from asyncio import all_tasks, as_completed
import math
import os
import time
import argparse
import random
from os.path import join
import torch
import torchvision
import pandas as pd

import numpy as np

from simclr import SimCLR

from linear_classifier import LinearClassifier
from simclr.modules.transformations import TransformsSimCLR

from utils import yaml_config_hook
from fs_dataset import FSDataset
import torch.nn.functional as F
import torchvision.models as torchvision_models
import vits
from simclr.modules.identity import Identity

def entropy(p):
    e = 0.0
    for p_i in p:
        e += p_i * math.log(p_i)
    return -e

def inference(loader, model, device):
    #print(f"=> Compute feature 0.00%. ", end='')
    begin = time.time()
    feature_vector = []
    labels_vector = []
    for step, (x, y) in enumerate(loader):
        x = x.to(device)
        with torch.no_grad():
            h, _, z, _ = model(x, x)

        h = h.detach()
        feature_vector.extend(h.cpu().detach().numpy())
        labels_vector.extend(y.numpy())

        print(end='\r')
        print(f"=> Compute feature {(100 * (step+1) / len(loader)):.2f}%. ", end='')

    feature_vector = np.array(feature_vector)
    labels_vector = np.array(labels_vector)
    end = time.time()
    print(f"{(end-begin)/60:.2f}min")
    return feature_vector, labels_vector

def create_data_loaders_from_arrays(X_train, y_train, X_test, y_test, batch_size):
    train = torch.utils.data.TensorDataset(
        torch.from_numpy(X_train), torch.from_numpy(y_train)
    )
    train_loader = torch.utils.data.DataLoader(
        train, batch_size=batch_size, shuffle=False
    )

    test = torch.utils.data.TensorDataset(
        torch.from_numpy(X_test), torch.from_numpy(y_test)
    )
    test_loader = torch.utils.data.DataLoader(
        test, batch_size=batch_size, shuffle=False
    )
    return train_loader, test_loader

def train(args, loader, classifier, criterion, optimizer, query_loader):
    loss_epoch = 0
    accuracy_epoch = 0

    for step, (x, y) in enumerate(loader):
        optimizer.zero_grad()

        x = x.to(args.device)
        y = y.to(args.device)

        output = classifier(x)
        loss = criterion(output, y)


        loss_entropy = 0
        for cnt, (x_q, y_q) in enumerate(query_loader):
            if cnt > args.entropy_cnt:
                break
            X_q = X_q.to(args.device)
            output_q = classifier(x_q)
            loss_entropy += entropy(output_q)
        loss_entropy /= args.entropy_cnt

        acc = (output.argmax() == y.argmax()).sum().item() / y.size(0)
        accuracy_epoch += acc

        loss.backward()
        optimizer.step()

        loss_epoch += loss.item()

    return loss_epoch/len(loader), accuracy_epoch/len(loader)

def regularization(args, loader, classifier, criterion, optimizer):
    loss_epoch = 0
    accuracy_epoch = 0
    for step, (x, y) in enumerate(loader):
        optimizer.zero_grad()

        x = x.to(args.device)
        y = y.to(args.device)

        output = classifier(x)

        loss = 0.0001 * entropy(output[0])
        acc = (output.argmax() == y.argmax()).sum().item() / y.size(0)
        accuracy_epoch += acc

        loss.backward()
        optimizer.step()

        loss_epoch += loss.item()

    return loss_epoch/len(loader), accuracy_epoch/len(loader)

def test(args, loader, classifier, criterion, optimizer):
    loss_epoch = 0
    accuracy_epoch = 0
    classifier.eval()
    for step, (x, y) in enumerate(loader):
        classifier.zero_grad()

        x = x.to(args.device)
        y = y.to(args.device)

        output = classifier(x)
        loss = criterion(output, y)

        acc = (output.argmax() == y.argmax()).sum().item() / y.size(0)
        accuracy_epoch += acc

        loss_epoch += loss.item()

    return loss_epoch/len(loader), accuracy_epoch/len(loader)

def generate_csv(dataset):
    dataset_path = join("..\\..\\dataset", dataset)
    subfolders = [
        # "train",
        #"test",
        "val"
        ]
    fn = []
    ln = []
    label_cnt = 0
    for subfoler in subfolders:
        path = join(dataset_path, subfoler)
        classes = os.listdir(path)
        for cls in classes:
            cls_path = join(path, cls)
            images = os.listdir(cls_path)
            for img in images:
                fn.append(join(path, join(cls, img)))
                ln.append(label_cnt)
            label_cnt += 1
    csv_fn = join(RESULT_FOLDER, f"{dataset}_{TIME_FN}.csv")
    pd.DataFrame({ "path":fn, "label":ln}).to_csv(csv_fn, header=True, index=False)
    return label_cnt, csv_fn

def sample_from_loader(all_features, args):
    #
    sample_way = random.sample(range(0, args.n_classes), args.way)
    index_mapping = {}
    for i in range(len(sample_way)):
        index_mapping[sample_way[i]] = i
    #
    shot_cnt = [0] * args.way
    support_X_list = []
    support_y_list = []
    query_X_list = []
    query_y_list = []

    for i in range(len(all_features[0])):
        X = all_features[0][i]
        y = all_features[1][i]
        if y in  sample_way:
            tmp_label = [0.0] * args.way
            tmp_label[index_mapping[y]] = 1.0
            if shot_cnt[index_mapping[y]] < args.shot:
                support_X_list.append(X)
                support_y_list.append(tmp_label)
                shot_cnt[index_mapping[y]] += 1
            else:
                query_X_list.append(X)
                query_y_list.append(tmp_label)

    support = torch.utils.data.TensorDataset(
        torch.from_numpy(np.array(support_X_list)), torch.from_numpy(np.array(support_y_list))
    )
    query = torch.utils.data.TensorDataset(
        torch.from_numpy(np.array(query_X_list)), torch.from_numpy(np.array(query_y_list))
    )
    arr_support_loader = torch.utils.data.DataLoader(
        support, batch_size=args.logistic_batch_size, shuffle=True,
    )
    arr_query_loader = torch.utils.data.DataLoader(
        query, batch_size=args.logistic_batch_size, shuffle=True
    )
    return arr_support_loader, arr_query_loader, sample_way


def worker(args, all_features):
    begin = time.time()
    sample_acc = 0.0
    for epi in range(0, 3):
        #
        arr_support_loader, arr_query_loader, sample_way = sample_from_loader(all_features, args)
        if args.support_init:
            mean_support_feature = torch.zeros(args.way, args.n_features)
            for step, (x, y) in enumerate(arr_support_loader):
                mean_support_feature[y.argmax()] += x[0]
            n_shot = len(arr_support_loader) / args.way
            mean_support_feature = F.normalize(mean_support_feature / n_shot)
        else:
            mean_support_feature = None
        #
        # ------------------------------------------------------------------------
        # Define classifier list
        # ------------------------------------------------------------------------
        classifier_list = []
        entropy_cnt_list = [0, 1, 5, 10, 20, 50]
        for i in range(len(entropy_cnt_list)):
            classifier= LinearClassifier(args.n_features, args.way, weight=mean_support_feature)
            classifier = classifier.to(args.device)
            optimizer = torch.optim.Adam(classifier.parameters(), lr=3e-4)
            criterion = torch.nn.CrossEntropyLoss()
            args.entropy_cnt = entropy_cnt_list[i]
            # ------------------------------------------------------------------------
            # Fine-tuning
            # ------------------------------------------------------------------------
            plot_x = [i for i in range(args.logistic_epochs)]
            plot_acc_y = []
            plot_loss_y = []
            for epoch in range(args.logistic_epochs):
                loss_epoch, accuracy_epoch = train(args, arr_support_loader, classifier, criterion, optimizer, arr_query_loader)
                plot_acc_y.append(accuracy_epoch)
                plot_loss_y.append(loss_epoch)
            # ------------------------------------------------------------------------
            # Testing
            # ------------------------------------------------------------------------
            loss_epoch, accuracy_epoch = test(args, arr_query_loader, classifier, criterion, optimizer)
            # ------------------------------------------------------------------------
            # Save 
            # ------------------------------------------------------------------------
            tmp = {}
            tmp["plot_x"] = plot_x
            tmp["plot_acc_y"] = plot_acc_y
            tmp["plot_loss_y"] = plot_loss_y
            tmp["entropy_cnt"] = args.entropy_cnt
            tmp["acc"] = accuracy_epoch
            tmp["loss"] = loss_epoch

            classifier_list.append(tmp)
        
        def plot_list(classifier_list, label='acc'):
            import matplotlib.pyplot as plt
            plt.title(f'{args.dataset_name} {args.encoder} {args.way}way-{args.shot}shot')
            for i in range(len(entropy_cnt_list)):
                x  = classifier_list[i]["ploy_x"]
                y  = classifier_list[i][f"ploy_{label}_y"]
                test_res = classifier_list[i][label]
                entropy_cnt = classifier_list[i]["entropy_cnt"]

                plt.plot(np.array(x), np.array(y), '^-', label=f"q={entropy_cnt}, test {label}={test_res:.4f})")
                if label == 'acc':
                    plt.ylim(0.0, 1.0)
                plt.xlim(0, x[-1]+1)
                plt.xlabel('epoch')
                plt.ylabel(label)
                plt.legend()
                plt.savefig(join(RESULT_FOLDER, f"{args.way}way-{args.shot}shot_{label}_{epi}.jpg"))
                plt.cla()
        plot_list(classifier_list, "acc")
        plot_list(classifier_list, "loss")
        # # ------------------------------------------------------------------------
        # # Define classifier
        # # ------------------------------------------------------------------------
        # classifier = LinearClassifier(args.n_features, args.way, weight=mean_support_feature)
        # #
        # classifier = classifier.to(args.device)
        # optimizer = torch.optim.Adam(classifier.parameters(), lr=3e-4)
        # criterion = torch.nn.CrossEntropyLoss()
        # # ------------------------------------------------------------------------
        # # Fine-tuning
        # # ------------------------------------------------------------------------
        # for epoch in range(args.logistic_epochs):
        #     args.entropy = True
        #     loss_epoch, accuracy_epoch = train(args, arr_support_loader, classifier, criterion, optimizer)
        # # ------------------------------------------------------------------------
        # # Testing
        # # ------------------------------------------------------------------------
        # loss_epoch, accuracy_epoch = test(args, arr_query_loader, classifier, criterion, optimizer)

        sample_acc += classifier_list[3]["acc"]
        #sample_acc += accuracy_epoch
    sample_acc /= args.sample_epoch
    end = time.time()
    print(f"{args.dataset_name} {args.arch} {args.encoder} {args.way}way {args.shot}shot {sample_acc:.6f} {end-begin:.2f}s")
    return sample_acc

if __name__ == "__main__" :
    parser = argparse.ArgumentParser(description="SimCLR")
    config = yaml_config_hook(".\\config\\pred.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))

    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------------------
    # record cfg
    # ------------------------------------------------------------------------

    dataset_list = [
       #"mini-imagenet",
        "FC100",
        # "tiered_imagenet",
    ]

    model_list = [
        # arch       encoder        path
        # ["mocov3", "vit_base", "..\\..\\moco-v3\\vit-b-300ep.pth.tar"],
        ["mocov3", "vit_small", "..\\..\\moco-v3\\vit-s-300ep.pth.tar"],
        # #["mocov3", "resnet50", "..\\..\\moco-v3\\r-50-100ep.pth.tar"],
        # ["mocov3", "resnet50", "..\\..\\moco-v3\\r-50-300ep.pth.tar"],
        # #["mocov3", "resnet50", "..\\..\\moco-v3\\r-50-1000ep.pth.tar"],
        # ["resnet", "resnet18",""],
        # ["resnet", "resnet50",""],
        # ["simclr", "resnet18", "r-18-checkpoint_100.tar"],
        # ["simclr", "resnet50", "r-50-checkpoint_100.tar"],
    ]

    way_list = [2, 5, 10]
    shot_list = [
        1, 5, 10, 15, 20, 30, 50,70,100]

    # make dir
    TIME_FN = f"{time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())}"
    RESULT_FOLDER = join("result", f"{args.fn}_{TIME_FN}")
    os.mkdir(RESULT_FOLDER)
   

    # ------------------------------------------------------------------------
    # Loading data
    # ------------------------------------------------------------------------
    begin = time.time()
    dataset_dict = {}
    for dataset_ins in dataset_list:
        args.dataset_name = dataset_ins
        args.n_classes, csv_fn = generate_csv(dataset=args.dataset_name)
        #
        dataset = FSDataset(
                annotations_file = csv_fn,
                n_classes= args.n_classes,
                transform=TransformsSimCLR(size=args.image_size).test_transform,
                target_transform=None
        )
        dataset_dict[args.dataset_name] = torch.utils.data.DataLoader(
            dataset, batch_size=args.batch_size,
            shuffle=False, drop_last=False, num_workers=args.workers)

    end = time.time()
    print(f"=> Loading datasets done. {end-begin:.2f}s ")

    # ------------------------------------------------------------------------
    # Loading model
    # ------------------------------------------------------------------------
    begin = time.time()
    model_dict = {}
    for model_ins in model_list:
        args.arch = model_ins[0]
        args.encoder = model_ins[1]
        args.model_path = model_ins[2]

        if args.arch == "resnet":
            if args.encoder == "resnet18":
                encoder =  torchvision.models.resnet18(pretrained=True)
            elif args.encoder == "resnet50":
                encoder =  torchvision.models.resnet50(pretrained=True)
        elif args.arch == "simclr":
            if args.encoder == "resnet18":
                encoder =  torchvision.models.resnet18(pretrained=False)
            elif args.encoder == "resnet50":
                encoder =  torchvision.models.resnet50(pretrained=False)
        elif args.arch == "mocov3":
            if args.encoder.startswith('vit'):
                encoder = vits.__dict__[args.encoder]()
                linear_keyword = 'head'
            elif args.encoder == "resnet50":
                encoder = torchvision_models.__dict__[args.encoder]()
                linear_keyword = 'fc'
            else:
                raise NotImplementedError
            checkpoint = torch.load(args.model_path, map_location="cpu")
            state_dict = checkpoint['state_dict']
            for k in list(state_dict.keys()):
                # retain only base_encoder up to before the embedding layer
                if k.startswith('module.base_encoder') and not k.startswith('module.base_encoder.%s' % linear_keyword):
                    # remove prefix
                    state_dict[k[len("module.base_encoder."):]] = state_dict[k]
                # delete renamed or unused k
                del state_dict[k]
            encoder.load_state_dict(state_dict, strict=False)
        else:
            raise NotImplementedError
        if args.encoder.startswith('vit'):
            n_features = encoder.head.in_features
            encoder.head = Identity()
            args.image_size = 224
        else:
            n_features = encoder.fc.in_features  # get dimensions of fc layer
            encoder.fc = Identity()
        # load pre-trained model from checkpoint
        mn = f"{args.arch}-{args.encoder}"
        model_dict[mn] = SimCLR(encoder, args.projection_dim, n_features)
        if args.arch == 'simclr':
            model_dict[mn].load_state_dict(torch.load(args.model_path, map_location=args.device.type))
    end = time.time()
    print(f"=> Loading models done. {end-begin:.2f}s")

    with open(join(RESULT_FOLDER, "cfg"), 'w') as f:
       f.write(f"dataset_list:{dataset_list}\n")
       f.write(f"model_list:{model_list}\n")
       f.write(f"way_list:{way_list}\n")
       f.write(f"shot_list:{shot_list}\n\n")
       
       f.write("\nyaml:")
       for k in args.__dict__:
           f.write(f"{k}: {args.__dict__[k]}\n")

    res_dict = {}
    for dataset_ins in dataset_list:
        args.dataset_name = dataset_ins
        dataloader = dataset_dict[args.dataset_name]
        for model_ins in model_list:
            args.arch = model_ins[0]
            args.encoder = model_ins[1]
            args.model_path = model_ins[2]
            mn = f"{args.arch}-{args.encoder}"
            model = model_dict[mn]
            model = model.to(args.device)
            model.eval()
            args.n_features = model.n_features
            # ------------------------------------------------------------------------
            # Get feature
            # ------------------------------------------------------------------------
            X, y = inference(dataloader, model, args.device)
            all_features = (X, y)
            print("feature compute done.")
            # ------------------------------------------------------------------------
            # M-way, N-shot
            # ------------------------------------------------------------------------
            for way_ins in way_list:
                for shot_ins in shot_list:
                    args.way = way_ins
                    args.shot = shot_ins
                    res_dict[f"{dataset_ins}-{model_ins[0]}-{model_ins[1]}-{way_ins}way-{shot_ins}shot"]= worker(args, all_features)
                
    print("=> Begining write results ...")
    with open(join(RESULT_FOLDER, f"{TIME_FN}.txt"), 'w') as f:
        for dataset_ins in dataset_list:
            for model_ins in model_list:
                for way_ins in way_list:
                    for shot_ins in shot_list:
                        key = f"{dataset_ins}-{model_ins[0]}-{model_ins[1]}-{way_ins}way-{shot_ins}shot"
                        f.write(f"{key} {res_dict[key]}\n")
    print(f"=> Writing results done.")

    import matplotlib.pyplot as plt
    import numpy as np

    for dataset_ins in dataset_list:
        for model_ins in model_list:
            for way_ins in way_list:
                res_dict[f"{dataset_ins}-{model_ins[0]}-{model_ins[1]}-{way_ins}way-0shot"] = 1.0/way_ins
    shot_list.insert(0, 0)
    for dataset_ins in dataset_list:
        for way_ins in way_list:
            for model_ins in model_list:
                y = []
                for shot_ins in shot_list:
                    key = f"{dataset_ins}-{model_ins[0]}-{model_ins[1]}-{way_ins}way-{shot_ins}shot"
                    y.append(res_dict[key])
                y = np.array(y)
                plt.plot(np.array(shot_list), y, '^-', label=f"{model_ins[0]}-{model_ins[1]}")
            plt.title(f'{way_ins}-way,n-shot ({dataset_ins})')
            plt.ylim(0.0, 1.0)
            plt.xlim(0, shot_list[-1])
            plt.xlabel('shot')
            plt.ylabel('accuracy')
            plt.legend()
            plt.savefig(join(RESULT_FOLDER, f"{dataset_ins}_{way_ins}way_nshot_randInit_{TIME_FN}.jpg"))
            plt.cla()

    with open(join(RESULT_FOLDER, "Done"), 'w') as f:
        pass