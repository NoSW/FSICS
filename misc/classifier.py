import math
import os
import time
import argparse
import random
import json
from os.path import join
import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt

from linear_classifier import LinearClassifier
from utils import yaml_config_hook

RESULT_FOLDER = ""
plt.style.use(['science', 'grid','notebook','no-latex'])

def mixup(args, loader, classifier, criterion, optimizer):
    for step1, (x1, y1) in enumerate(loader):
        x1 = x1.to(args.device)
        y1 = y1.to(args.device)
        for step2, (x2, y2) in enumerate(loader):
            x2 = x2.to(args.device)
            y2 = y2.to(args.device)
            if y1.argmax(1) != y2.argmax(1):
                lam = np.random.beta(0.3, 0.3)
                x = lam * x1 + (1 - lam) * x2
                y = lam * y1 + (1 - lam) * y2
                optimizer.zero_grad()
                output = classifier(x)
                loss = criterion(output, y)
                loss.backward()
                optimizer.step()

def train(args, loader, classifier, criterion, optimizer):
    loss_epoch = 0
    accuracy_epoch = 0
    for step, (x, y) in enumerate(loader):
        optimizer.zero_grad()

        x = x.to(args.device)
        y = y.to(args.device)

        output = classifier(x)
        loss = criterion(output, y)

        acc = (output.argmax(1) == y.argmax(1)).sum().item() / y.size(0)
        accuracy_epoch += acc

        loss.backward()
        optimizer.step()

        loss_epoch += loss.item()

    return loss_epoch/len(loader), accuracy_epoch/len(loader)

def regularization(args, loader, classifier, optimizer):
    def entropy(output):
        res = 0
        for p in output:
            e = 0.0
            for p_i in p:
                e += p_i * math.log(p_i)
            res += -e
        return res / output.shape[0]

    loss = 0
    optimizer.zero_grad()
    for _, (x, y) in enumerate(loader):
        x = x.to(args.device)
        output = classifier(x)
        loss += entropy(output)
    loss /= len(loader)
    loss.backward()
    optimizer.step()

def zero_loss(args, classifier, optimizer):
    for _ in range(300):
        optimizer.zero_grad()
        loss = classifier(torch.Tensor([[0.1]*args.n_features]).to(args.device))[0][0] * 0
        loss.backward()
        optimizer.step()

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

        acc = (output.argmax(1) == y.argmax(1)).sum().item() / y.size(0)
        accuracy_epoch += acc

        loss_epoch += loss.item()

    return loss_epoch/len(loader), accuracy_epoch/len(loader)

def sample_from_loader(X, args):
    sample_way = random.sample(range(0, args.n_classes), args.way)

    support_X_list = []
    support_y_list = []
    query_X_list = []
    query_y_list = []
    for way_ins in sample_way:
        tmp_label = [0.0] * args.way
        tmp_label[sample_way.index(way_ins)] = 1.0
        for shot_ins in range(args.sample_per_class):
            if shot_ins >= args.shot:
                query_X_list.append(X[way_ins * args.sample_per_class + shot_ins])
                query_y_list.append(tmp_label)
            else:
                support_X_list.append(X[way_ins * args.sample_per_class + shot_ins])
                support_y_list.append(tmp_label)


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
        query, batch_size=(args.logistic_batch_size * 8), shuffle=True
    )
    return arr_support_loader, arr_query_loader

def get_mean_support_feature(args, arr_support_loader):
    mean_support_feature = torch.zeros(args.way, args.n_features)
    for step, (x, y) in enumerate(arr_support_loader):
        mean_support_feature[y.argmax()] += x[0]
    mean_support_feature = F.normalize(mean_support_feature / args.shot)
    return mean_support_feature

def plot_list(args, classifier_list, label, save_as):
    print(f"ploting {args.way} {args.shot} {len(classifier_list)}")
    plt.title(f'{args.dataset} {args.encoder} {args.way}way-{args.shot}shot')
    for i in range(args.entropy_corr_epoch):
        x  = classifier_list[i]["plot_x"]
        y  = classifier_list[i][f"plot_{label}_y"]
        test_res = classifier_list[i][label]
        entropy_corr = classifier_list[i]["entropy_corr"]
        plt.plot(np.array(x), np.array(y), '^-', label=f"{entropy_corr:.0e}, {label}={test_res:.4f})")
    if label == 'acc':
        plt.ylim(0.0, 1.0)
    plt.xlabel('epoch')
    plt.ylabel(label)
    plt.legend()
    plt.savefig(save_as)
    plt.cla()

def entropy_corr_probing(args, arr_support_loader, arr_query_loader, mean_support_feature,epi ):
    # ------------------------------------------------------------------------
        # Define classifier list
        # ------------------------------------------------------------------------
        entropy_corr_list = [1000, 100, 10, 1, 1e-10, 1e-50, 1e-100,0, 1e-10000000, 1e-1000000000, 1e-400, 1e-70]
        classifier_list = []
        entropy_corr_idx = entropy_corr_list.index(1e-50)
        entropy_list = []
        becoming_big = True
        for i in range(args.entropy_corr_epoch):
            begin = time.time()
            classifier= LinearClassifier(args.n_features, args.way, weight=mean_support_feature)
            classifier = classifier.to(args.device)
            optimizer = torch.optim.Adam(classifier.parameters(), lr=3e-4)
            criterion = torch.nn.CrossEntropyLoss()
            args.entropy_corr = entropy_corr_list[entropy_corr_idx ] if i < args.entropy_corr_epoch - 1 else 0
            entropy_list.append(entropy_corr_idx)
            print(f"i={i} {args.way} {args.shot} {args.entropy_corr}", end ='\r')
            # ------------------------------------------------------------------------
            # Fine-tuning
            # ------------------------------------------------------------------------
            plot_x = []
            plot_acc_y = []
            plot_loss_y = []
            for epoch in range(args.logistic_epochs*20):
                loss_epoch, accuracy_epoch = train(args, arr_support_loader, classifier, criterion, optimizer)
                if i < args.entropy_corr_epoch - 1:
                   loss_epoch_r, accuracy_epoch_r = regularization(args, arr_query_loader, classifier, optimizer)
                if epoch % args.plot_sample_rate == 0:
                    plot_x.append(epoch)
                    plot_acc_y.append(accuracy_epoch)
                    plot_loss_y.append(loss_epoch)
            # ------------------------------------------------------------------------
            # Testing
            # ------------------------------------------------------------------------
            loss_epoch, accuracy_epoch = test(args, arr_query_loader, classifier, criterion, optimizer)
            # ------------------------------------------------------------------------
            # Save state
            # ------------------------------------------------------------------------
            classifier_list.append({
                "plot_x": plot_x,
                "plot_acc_y": plot_acc_y,
                "plot_loss_y": plot_loss_y,
                "entropy_corr": args.entropy_corr,
                "acc": accuracy_epoch,
                "loss": loss_epoch
            })
            # ------------------------------------------------------------------------
            # Adjust entropy corr
            # ------------------------------------------------------------------------
            if i < args.entropy_corr_epoch - 2:
                if  classifier_list[i]['acc'] < max(0, classifier_list[i-1]['acc'] - 0.02):
                    becoming_big = not becoming_big
                while entropy_corr_idx in entropy_list:
                    entropy_corr_idx += (-1 if becoming_big else 1)
            print(f"i={i} {args.way} {args.shot} {entropy_corr_list[entropy_corr_idx ]}(next) {time.time()-begin:.4f}s")   
        # ------------------------------------------------------------------------
        # Ploting
        # ------------------------------------------------------------------------
        for label in ['acc', 'loss']:
            plot_list(args, classifier_list, label,
                save_as=join(RESULT_FOLDER, f"{args.way}way-{args.shot}shot_{label}_{epi}.jpg"))

        return max([classifier_list[a]['acc'] for a in range(args.entropy_corr_epoch)])

def extra_loss(args, arr_support_loader, arr_query_loader, mean_support_feature,zero_enabled=False, mixup_enabled=False):
    classifier= LinearClassifier(args.n_features, args.way, weight=mean_support_feature)
    classifier = classifier.to(args.device)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=3e-4)
    criterion = torch.nn.CrossEntropyLoss()
    plot_x = []
    plot_acc_y = []
    plot_loss_y = []
    plot_acc_t = []
    for epoch in range(args.logistic_epochs):
        loss_epoch, accuracy_epoch = train(args, arr_support_loader, classifier, criterion, optimizer)
        if zero_enabled:
            zero_loss(args, classifier, optimizer)
            #regularization(args,arr_query_loader, classifier, optimizer)
        if mixup_enabled:
            mixup(args, arr_support_loader, classifier, criterion, optimizer)

        if epoch % args.plot_sample_rate == 0:
            plot_x.append(epoch)
            plot_acc_y.append(accuracy_epoch)
            plot_loss_y.append(loss_epoch)
            loss_epoch, accuracy_epoch = test(args, arr_query_loader, classifier, criterion, optimizer)
            plot_acc_t.append(accuracy_epoch)
            print(f"[{epoch}/{args.logistic_epochs} test acc:{accuracy_epoch}]") 
    return plot_x, plot_acc_y, plot_loss_y, accuracy_epoch, loss_epoch, plot_acc_t

def two_com(args, arr_support_loader, arr_query_loader, mean_support_feature, epi):
    x, acc_mixup, loss_mixup, acc_mixup_t, loss_mixup_t, plot_mixup_t = extra_loss(args, arr_support_loader, arr_query_loader, mean_support_feature, zero_enabled=False, mixup_enabled=True)
    print("DONE 1.")
    x, acc_blank, loss_blank, acc_blank_t, loss_blank_t, plot_blank_t  = extra_loss(args, arr_support_loader, arr_query_loader, mean_support_feature, zero_enabled=False, mixup_enabled=False)
    print("DONE 2.")
    x, acc_zero, loss_zero, acc_zero_t, loss_zero_t, plot_zero_t  = extra_loss(args, arr_support_loader, arr_query_loader, mean_support_feature, zero_enabled=True, mixup_enabled=False)
    print("DONE 3.")
   # x, acc_both, loss_both, acc_both_t, loss_both_t, plot_both_t  = extra_loss(args, arr_support_loader, arr_query_loader, mean_support_feature, zero_enabled=True, mixup_enabled=True)
   # print("DONE 4.")
    def plot_two_sum(args, d, line, la):
        for l in line:
            t = d[f"{la}_{l}_t"]
            plt.plot(np.array(d['x']), np.array(d[f"{la}_{l}"]), '^-', label=f"{l} {la}={t:.4f})")
        plt.title(f'{args.dataset} {args.encoder} {args.way}way-{args.shot}shot')
        plt.xlabel('epoch')
        plt.ylabel(la)
        plt.legend()
        plt.savefig(join(RESULT_FOLDER, f"{args.dataset} {args.encoder} {args.way}way-{args.shot}shot_{la}-{epi}.png"))
        plt.cla()

    fn = join(RESULT_FOLDER, f"{args.dataset}-{args.encoder}-{args.way}way-{args.shot}shot-{epi}.json")
    res_dict = {
        'x': x ,
        "acc_blank": acc_blank, "acc_blank_t": acc_blank_t, "loss_blank": loss_blank, "loss_blank_t": loss_blank_t, "plot_blank_t": plot_blank_t, 
        "acc_zero": acc_zero, "acc_zero_t": acc_zero_t, "loss_zero": loss_zero, "loss_zero_t": loss_zero_t, "plot_zero_t": plot_zero_t,
        "acc_mixup": acc_mixup, "acc_mixup_t": acc_mixup_t, "loss_mixup": loss_mixup, "loss_mixup_t": loss_mixup_t, "plot_mixup_t": plot_mixup_t,
       # "acc_both": acc_both, "acc_both_t": acc_both_t, "loss_both": loss_both, "loss_both_t": loss_both_t, "plot_both_t": plot_both_t,
    }
    with open(fn, 'w') as f:
            json.dump(res_dict,f)

    #plot_two_sum(args, res_dict, ["blank", "zero", "mixup", "both"], 'acc')
    #plot_two_sum(args, res_dict, ["blank", "zero", "mixup", "both"], 'loss')

    return acc_blank_t

def instance(args, arr_support_loader, arr_query_loader, mean_support_feature, epi):
    classifier = LinearClassifier(args.n_features, args.way, weight=mean_support_feature)
    classifier = classifier.to(args.device)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=3e-4)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(args.logistic_epochs):
        loss_epoch, accuracy_epoch = train(args, arr_support_loader, classifier, criterion, optimizer)
        if args.entropy_enabled:
            loss_epoch_r, accuracy_epoch_r = regularization(args, arr_query_loader, classifier, optimizer)
    loss_epoch, accuracy_epoch = test(args, arr_query_loader, classifier, criterion, optimizer)
    
    return accuracy_epoch

def worker(args, X):
    begin = time.time()
    sample_acc = 0.0
    for epi in range(args.sample_epoch):
        print(epi, "epi")
        arr_support_loader, arr_query_loader = sample_from_loader(X, args)
        mean_support_feature = get_mean_support_feature(args, arr_support_loader) \
            if args.support_init_enabled else None
        sample_acc += two_com(args, arr_support_loader, arr_query_loader, mean_support_feature, epi)
        #sample_acc += instance(args, arr_support_loader, arr_query_loader, mean_support_feature, epi)
    sample_acc /= args.sample_epoch
    print(f"{args.dataset} {args.arch} {args.encoder} {args.way}way {args.shot}shot {sample_acc:.6f} {time.time()-begin:.2f}s")
    return sample_acc

if __name__ == "__main__" :
    # ------------------------------------------------------------------------
    # init
    # ------------------------------------------------------------------------
    parser = argparse.ArgumentParser(description="SimCLR")
    config = yaml_config_hook(".\\config\\classifier.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))
    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # ------------------------------------------------------------------------
    # Setting
    # ------------------------------------------------------------------------
    dataset_list = [
      #  "mini-imagenet",
        "FC100",
    ]
    model_list = [
        # arch       encoder      
        # ["mocov3", "vit_base"],
        ["mocov3", "vit_small",],
        # ["mocov3", "resnet50"],
        # ["resnet", "resnet50"],
    ]
    way_shot_dict ={
        # way [shot0, shot1, ...]
        2: [1, 5, 10, 20],
        5: [5],#[1, 5, 10, 20],
        10:[10],# [1, 5, 10, 20],
        50: [5,],#[1, 5, 10, 20],
    }
    # ------------------------------------------------------------------------
    # Make dir
    # ------------------------------------------------------------------------
    TIME_FN = f"{time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())}-{random.randint(0,10000)}"
    RESULT_FOLDER = join("result", f"{args.fn}_{TIME_FN}")
    os.mkdir(RESULT_FOLDER)
    with open(join(RESULT_FOLDER, "acfg"), 'w') as f:
        f.write(f"{args.desc}\n\n")
        f.write(f"dataset_list:{dataset_list}\n")
        f.write(f"model_list:{model_list}\n")
        f.write(f"way_shot:\n")
        for k in way_shot_dict:
            f.write(f"{k}: {way_shot_dict[k]}")

        f.write("\nyaml:")
        for k in args.__dict__:
            if k != 'desc':
                f.write(f"{k}: {args.__dict__[k]}\n")
    # ------------------------------------------------------------------------
    # Run DxExN iters
    # ------------------------------------------------------------------------
    res_dict = {}
    for dataset_ins in dataset_list:
        args.dataset = dataset_ins
        for model_ins in model_list:
            args.arch = model_ins[0]
            args.encoder = model_ins[1]
            # ------------------------------------------------------------------------
            # Get feature
            # ------------------------------------------------------------------------
            X = np.load(join("feature", f"{args.dataset}-{args.arch}-{args.encoder}.npy"))
            args.n_features = X.shape[1]
            print(f"feature read done: X.shape={X.shape}.")
            # ------------------------------------------------------------------------
            # M-way, N-shot
            # ------------------------------------------------------------------------
            for shot_ins in way_shot_dict[args.way]:
                args.shot = shot_ins
                res_dict[f"{dataset_ins}-{model_ins[0]}-{model_ins[1]}-{args.way}way-{shot_ins}shot"]= worker(args, X)
    # ------------------------------------------------------------------------
    # Writing result
    # ------------------------------------------------------------------------           
    print("=> Begining write results ...")
    with open(join(RESULT_FOLDER, f"a-result-{TIME_FN}.txt"), 'w') as f:
        for dataset_ins in dataset_list:
            for model_ins in model_list:
                for shot_ins in way_shot_dict[args.way]:
                    key = f"{dataset_ins}-{model_ins[0]}-{model_ins[1]}-{args.way}way-{shot_ins}shot"
                    f.write(f"{key} {res_dict[key]}\n")