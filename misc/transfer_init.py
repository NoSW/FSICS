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

def instance(args, epi):
    new_shot = args.shot
    old_index = way_shot_dict[args.way].index(new_shot) + 1
    args.shot = way_shot_dict[args.way][min(old_index, len(way_shot_dict[args.way])-1)]
    arr_support_loader, arr_query_loader = sample_from_loader(X, args)
    print(f"len(supp):{len(arr_support_loader)}")
    #mean_support_feature = get_mean_support_feature(args, arr_support_loader) \
    #     if args.support_init_enabled else None
    classifier = LinearClassifier(args.n_features, args.way, weight=None)
    classifier = classifier.to(args.device)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=3e-4)
    criterion = torch.nn.CrossEntropyLoss()

    plot_x = []
    old_acc_y = []
    old_loss_y = []
    for epoch in range(args.logistic_epochs):
        loss_epoch, accuracy_epoch = train(args, arr_support_loader, classifier, criterion, optimizer)
        if epoch % args.plot_sample_rate == 0:
            plot_x.append(epoch)
            old_acc_y.append(accuracy_epoch)
            old_loss_y.append(loss_epoch)
    # loss_old, accuracy_old = test(args, arr_query_loader, classifier, criterion, optimizer)
    loss_old, accuracy_old = 0, 0

    new_acc_y = []
    new_loss_y = []
    args.shot = new_shot    
    arr_support_loader, arr_query_loader = sample_from_loader(X, args)
    print(f"len(supp):{len(arr_support_loader)}")
    for epoch in range(args.logistic_epochs):
        loss_epoch, accuracy_epoch = train(args, arr_support_loader, classifier, criterion, optimizer)
        if epoch % args.plot_sample_rate == 0:
            new_acc_y.append(accuracy_epoch)
            new_loss_y.append(loss_epoch)
    loss_new, accuracy_new = test(args, arr_query_loader, classifier, criterion, optimizer)


    with open(join(RESULT_FOLDER ,f"{args.dataset}-{args.way}way-{args.shot}shot-{epi}.json"), 'w') as f:
            json.dump({
                "plot_x": plot_x,
                "old_acc_y": old_acc_y,
                "old_loss_y": old_loss_y,
                "old_acc": accuracy_old,
                "old_loss": loss_old,
                "new_acc_y": new_acc_y,
                "new_loss_y": new_loss_y,
                "new_acc": accuracy_new,
                "new_loss": loss_new,
            },f)
    return accuracy_new

def worker(args, X):
    begin = time.time()
    sample_acc = 0.0
    for epi in range(args.sample_epoch):
        sample_acc += instance(args, epi)
    sample_acc /= args.sample_epoch
    print(f"{args.dataset} {args.arch} {args.encoder} {args.way}way {args.shot}shot {sample_acc:.6f} {time.time()-begin:.2f}s")
    return sample_acc

if __name__ == "__main__" :
    # ------------------------------------------------------------------------
    # init
    # ------------------------------------------------------------------------
    parser = argparse.ArgumentParser(description="support based initialization")
    config = yaml_config_hook(".\\config\\supp.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))
    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.way == 50:
        args.sample_epoch = 2
    # ------------------------------------------------------------------------
    # Setting
    # ------------------------------------------------------------------------
    dataset_list = [
        "mini-imagenet",
        "FC100",
    ]
    model_list = [
        # arch       encoder      
        # ["mocov3", "vit_base"],
        ["mocov3", "vit_small",],
        # #["mocov3", "resnet50"],
        # ["mocov3", "resnet50"],
        # #["mocov3", "resnet50"],
        # ["resnet", "resnet18",""],
        # ["resnet", "resnet50",""],
    ]
    way_shot_dict ={
        # way [shot0, shot1, ...]
        2: [1, 5, 10, 20],
        5: [1, 5, 10, 20],
        10: [1, 5, 10, 20],
        50: [1, 5, 10, 20],
    }
    # ------------------------------------------------------------------------
    # Make dir
    # ------------------------------------------------------------------------
    RESULT_FOLDER = join("result", f"{args.fn}")
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
    with open(join(RESULT_FOLDER, f"a-result.txt"), 'w') as f:
        for dataset_ins in dataset_list:
            for model_ins in model_list:
                for shot_ins in way_shot_dict[args.way]:
                    key = f"{dataset_ins}-{model_ins[0]}-{model_ins[1]}-{args.way}way-{shot_ins}shot"
                    f.write(f"{key} {res_dict[key]}\n")