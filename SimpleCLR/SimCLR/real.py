from asyncio import all_tasks, as_completed
import math
import os
import time
import logging
import argparse
import random
from os.path import join
from concurrent.futures import ThreadPoolExecutor, wait
import torch
import torchvision
import pandas as pd

import numpy as np

from simclr import SimCLR

from linear_classifier import LinearClassifier
from simclr.modules.transformations import TransformsSimCLR
from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn

from utils import yaml_config_hook
from torch.utils.data import DataLoader
from fs_dataset import FSDataset
import torch.nn.functional as F
import torchvision.models as torchvision_models
import vits
from simclr.modules.identity import Identity

logging.basicConfig(level=logging.INFO)

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

def train(args, loader, classifier, criterion, optimizer):
    loss_epoch = 0
    accuracy_epoch = 0
    for step, (x, y) in enumerate(loader):
        optimizer.zero_grad()

        x = x.to(args.device)
        y = y.to(args.device)

        output = classifier(x)

        loss = criterion(output, y)
        acc = (output.argmax() == y.argmax()).sum().item() / y.size(0)
        accuracy_epoch += acc

        loss.backward()
        optimizer.step()

        loss_epoch += loss.item()

    return loss_epoch, accuracy_epoch

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

    return loss_epoch, accuracy_epoch

def regular(args, loader, classifier, optimizer):
    loss_epoch = 0
    accuracy_epoch = 0
    for _, (x, y) in enumerate(loader):
        optimizer.zero_grad()

        x = x.to(args.device)
        y = y.to(args.device)

        output = classifier(x)

        loss_epoch += entropy(output[0])

        acc = (output.argmax() == y.argmax()).sum().item() / y.size(0)
        accuracy_epoch += acc
    loss_epoch /= len(loader)
    loss_epoch.backward()
    optimizer.step()

    return loss_epoch, accuracy_epoch

def generate_csv(dataset):
    dataset_path = join("..\\..\\dataset", dataset)
    subfolders = ["train"]
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
    pd.DataFrame({ "path":fn, "label":ln}).to_csv("{}.csv".format(dataset), header=True, index=False)
    return label_cnt

def generate_csv_debug(dataset, subfolders):
    dataset_path = join("..\\..\\dataset", dataset)
    #
    fn = []
    ln = []
    label_cnt = 0
    for subfoler in subfolders:
        path = join(dataset_path, subfoler)
        classes = os.listdir(path)
        for cls in classes:
            cls_path = join(path, cls)
            images = os.listdir(cls_path)
            img_cnt = 0
            for img in images:
                if img_cnt >= 200:
                    break
                fn.append(join(path, join(cls, img)))
                ln.append(label_cnt)
                img_cnt += 1
            label_cnt += 1
    pd.DataFrame({ "path":fn, "label":ln}).to_csv("{}.csv".format(dataset), header=True, index=False)
    return label_cnt

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
    for epi in range(0, args.sample_epoch):
        #
        arr_support_loader, arr_query_loader, sample_way = sample_from_loader(all_features, args)
        mean_support_feature = torch.zeros(args.way, args.n_features)
        for step, (x, y) in enumerate(arr_support_loader):
            mean_support_feature[y.argmax()] += x[0]
        n_shot = len(arr_support_loader) / args.way
        mean_support_feature = F.normalize(mean_support_feature / n_shot)
        #
        # ------------------------------------------------------------------------
        # Define classifier
        # ------------------------------------------------------------------------
        #
        classifier = LinearClassifier(args.n_features, args.way,
            weight=mean_support_feature, bias=torch.zeros(args.way))
        #
        classifier = classifier.to(args.device)
        optimizer = torch.optim.Adam(classifier.parameters(), lr=3e-4)
        criterion = torch.nn.CrossEntropyLoss()
        #
        # ------------------------------------------------------------------------
        # Fine-tuning
        # ------------------------------------------------------------------------
        for epoch in range(args.logistic_epochs):
            # loss_epoch, accuracy_epoch = regular(args, arr_query_loader, classifier,  optimizer)
            # print(
            #     f"Epoch [{epoch}/{args.logistic_epochs}]\t Loss: {loss_epoch / len(arr_query_loader)}\t Accuracy: {accuracy_epoch / len(arr_query_loader)}"
            # )
            loss_epoch, accuracy_epoch = train(
                args, arr_support_loader, classifier, criterion, optimizer
            )
            # print(
            #     f"Epoch [{epoch}/{args.logistic_epochs}]\t Loss: {loss_epoch / len(arr_support_loader)}\t Accuracy: {accuracy_epoch / len(arr_support_loader)}"
            # )
        # ------------------------------------------------------------------------
        # Entropy_regularization
        # ------------------------------------------------------------------------
        # for epoch in range(args.logistic_epochs):
        #     loss_epoch, accuracy_epoch = regular(args, arr_query_loader, classifier,  optimizer)
        #     print(
        #         f"Epoch [{epoch}/{args.logistic_epochs}]\t Loss: {loss_epoch / len(arr_query_loader)}\t Accuracy: {accuracy_epoch / len(arr_query_loader)}"
        #     )
        #
        # ------------------------------------------------------------------------
        # Testing
        # ------------------------------------------------------------------------
        loss_epoch, accuracy_epoch = test(args, arr_query_loader, classifier, criterion, optimizer)
        # print(
        #     f"[FINAL]\t Loss: {loss_epoch / len(arr_query_loader)}\t Accuracy: {accuracy_epoch / len(arr_query_loader)}"
        # )
        sample_acc += accuracy_epoch / len(arr_query_loader)
    sample_acc /= args.sample_epoch
    end = time.time()
    print(f"{args.dataset_name} {args.arch} {args.encoder} {args.way}way {args.shot}shot {sample_acc:.6f} {end-begin:.2f}s")
    return sample_acc
    #f"{args.dataset_name} {args.arch} {args.encoder} {args.way}way {args.shot}shot {sample_acc}"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SimCLR")
    config = yaml_config_hook("./config/pred.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))

    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.logistic_batch_size = 1

    # ------------------------------------------------------------------------
    # Loading data
    # ------------------------------------------------------------------------
    begin = time.time()
    dataset_list = [
        # "mini-imagenet",
        # "FC100",
        # "tiered_imagenet",
        "real_labeled",
    ]
    dataset_dict = {}
    for dataset_ins in dataset_list:
        args.dataset_name = dataset_ins
        args.n_classes = generate_csv(dataset=args.dataset_name)
        #
        dataset = FSDataset(
                annotations_file = "{}.csv".format(args.dataset_name),
                n_classes= args.n_classes,
                transform=TransformsSimCLR(size=args.image_size).train_transform,
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
    model_list = [
        # arch       encoder        path
        ["mocov3", "vit_base", "..\\..\\moco-v3\\vit-b-300ep.pth.tar"],
        # ["mocov3", "vit_small", "..\\..\\moco-v3\\vit-s-300ep.pth.tar"],
        # #["mocov3", "resnet50", "..\\..\\moco-v3\\r-50-100ep.pth.tar"],
        # ["mocov3", "resnet50", "..\\..\\moco-v3\\r-50-300ep.pth.tar"],
        # #["mocov3", "resnet50", "..\\..\\moco-v3\\r-50-1000ep.pth.tar"],
        # ["resnet", "resnet18",""],
        # ["resnet", "resnet50",""],
        # ["simclr", "resnet18", "r-18-checkpoint_100.tar"],
        # ["simclr", "resnet50", "r-50-checkpoint_100.tar"],
    ]
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
            # ------------------------------------------------------------------------
            # M-way, N-shot
            # ------------------------------------------------------------------------
            way_list = [3]
            shot_list = [1, 5, 10, 15, 20, 30, 50, 70, 100, 150, 200]
            for way_ins in way_list:
                for shot_ins in shot_list:
                    args.way = way_ins
                    args.shot = shot_ins
                    res_dict[f"{dataset_ins}-{model_ins[0]}-{model_ins[1]}-{way_ins}way-{shot_ins}shot"]= worker(args, all_features)
                
    print("=> Begining write results ...")
    with open('real_res.txt', 'w') as f:
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
            plt.savefig(f".\\fig\\{dataset_ins}_{way_ins}way_nshot.jpg")
            plt.cla()

