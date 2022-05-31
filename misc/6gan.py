import math
import os
import time
import argparse
import random
import json
from os.path import join
from turtle import Turtle
from black import out
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


def test(args, loader, classifier, criterion):
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

def train_pesudo(args, loader, classifier, criterion, optimizer, pred, epoch):
    loss_epoch = 0
    for _, (x, y) in enumerate(loader):
        optimizer.zero_grad()
        x = x.to(args.device)
        output1 = classifier(x)
        output2 = pred(x)
        if random.uniform(0, 1) < (0.002 * (epoch/50)*(epoch/50)):
            output2 = torch.tensor(np.random.dirichlet(np.ones(args.way), size=x.shape[0])).to(args.device)
        if epoch > 170:
            tmp = []
            idx = output2.argmax(1)
            for i in range(len(idx)):
                if output2[i][idx[i]] > 0.5:
                    tmptmp = [0.0] * args.way
                    tmptmp[idx[i]] = 1.0
                else:
                    tmptmp = []
                    for v in output2[i]:
                        tmptmp.append(v.item())
                tmp.append(tmptmp)
            output2 = torch.tensor(tmp).to(args.device)

        loss = criterion(output1, output2)
        loss.backward()
        optimizer.step()
        loss_epoch += loss.item()
    loss_epoch /= len(loader)
    return loss_epoch

def instance(args, arr_support_loader, arr_query_loader, mean_support_feature, epi):
    print(f"{args.dataset} {epi}")
    blank = LinearClassifier(args.n_features, args.way, weight=mean_support_feature)
    blank = blank.to(args.device)
    opt_blank = torch.optim.Adam(blank.parameters(), lr=3e-4)

    new = LinearClassifier(args.n_features, args.way, weight=mean_support_feature)
    new = new.to(args.device)
    opt_new = torch.optim.Adam(new.parameters(), lr=3e-4)

    old = LinearClassifier(args.n_features, args.way, weight=mean_support_feature)
    old = old.to(args.device)
    opt_old = torch.optim.Adam(old.parameters(), lr=3e-4)

    # backup = LinearClassifier(args.n_features, args.way, weight=mean_support_feature)
    # backup = backup.to(args.device)
    # opt_backup = torch.optim.Adam(backup.parameters(), lr=3e-4)


    # limit = LinearClassifier(args.n_features, args.way, weight=mean_support_feature)
    # limit = limit.to(args.device)
    # opt_limit = torch.optim.Adam(limit.parameters(), lr=3e-4)

    criterion = torch.nn.CrossEntropyLoss()

    plot_x = []
    plot_acc_blank = []
    plot_acc_old = []
    plot_acc_new = []
    flag = True
    for epoch in range(args.logistic_epochs):
        if epoch % 50 == 0:
            # if epoch % 100 == 0:
            #     backup, old = old, backup
            #     opt_backup, opt_old = opt_old, opt_backup
            # else:
            flag = not flag
            new ,old = old, new
            opt_new, opt_old = opt_old, opt_new

        # loss_epoch, accuracy_epoch = train(args, arr_support_loader, limit, criterion, opt_limit)
        # loss_epoch, accuracy_epoch = train(args, arr_query_loader, limit, criterion, opt_limit)
        # print(f"[{epoch}/{args.logistic_epochs}] {loss_epoch} {accuracy_epoch}(limit on query)")
        loss_epoch, accuracy_epoch = train(args, arr_support_loader, blank, criterion, opt_blank)
        print(f"[{epoch}/{args.logistic_epochs}] {loss_epoch} {accuracy_epoch}(blank on supp)")
        loss_epoch, accuracy_epoch = train(args, arr_support_loader, new, criterion, opt_new)
        print(f"[{epoch}/{args.logistic_epochs}] {loss_epoch} {accuracy_epoch}(new on supp)")
        # loss_epoch, accuracy_epoch = train(args, arr_support_loader, backup, criterion, opt_backup)
        # print(f"[{epoch}/{args.logistic_epochs}] {loss_epoch} {accuracy_epoch}(backup on supp)")

        loss_epoch = train_pesudo(args, arr_query_loader, new, criterion, opt_new, old, epoch)
        print(f"[{epoch}/{args.logistic_epochs}] {loss_epoch}(new on pesudo-query))")
        # loss_epoch = train_pesudo(args, arr_query_loader, backup, criterion, opt_backup, old, epoch)
        # print(f"[{epoch}/{args.logistic_epochs}] {loss_epoch}(backup on pesudo-query))")

        # test
        loss_epoch, accuracy_new = test(args, arr_query_loader, new, criterion)
        print(f"[{epoch}/{args.logistic_epochs}] {loss_epoch} {accuracy_new}(new on query)")
        # loss_epoch, accuracy_epoch = test(args, arr_query_loader, backup, criterion)
        # print(f"[{epoch}/{args.logistic_epochs}] {loss_epoch} {accuracy_epoch}(backup on query)")
        loss_epoch, accuracy_old = test(args, arr_query_loader, old, criterion)
        print(f"[{epoch}/{args.logistic_epochs}] {loss_epoch} {accuracy_old}(old on query)")
        loss_epoch, accuracy_blank = test(args, arr_query_loader, blank, criterion)
        print(f"[{epoch}/{args.logistic_epochs}] {loss_epoch} {accuracy_blank}(blank on query)")

        if epoch % args.plot_sample_rate == 0:
            plot_x.append(epoch)
            plot_acc_blank.append(accuracy_blank)    
            if flag:
                plot_acc_new.append(accuracy_new)    
                plot_acc_old.append(accuracy_old)    
            else:
                plot_acc_new.append(accuracy_old)
                plot_acc_old.append(accuracy_new)
        print("####################################")
    with open(f'{args.dataset}-luoxaun-{epi}.json', 'w') as f:
        json.dump({
            "plot_x": plot_x,
            "plot_acc_blank": plot_acc_blank,
            "plot_acc_old": plot_acc_old,
            "plot_acc_new": plot_acc_new,
        },f)
    return accuracy_epoch

def worker(args, X):
    begin = time.time()
    sample_acc = 0.0
    for epi in range(args.sample_epoch):
        print(epi, "epi")
        arr_support_loader, arr_query_loader = sample_from_loader(X, args)
        mean_support_feature = get_mean_support_feature(args, arr_support_loader) \
            if args.support_init_enabled else None
        sample_acc += instance(args, arr_support_loader, arr_query_loader, mean_support_feature, epi)
    sample_acc /= args.sample_epoch
    print(f"{args.dataset} {args.arch} {args.encoder} {args.way}way {args.shot}shot {sample_acc:.6f} {time.time()-begin:.2f}s")
    return sample_acc

if __name__ == "__main__" :
    # ------------------------------------------------------------------------
    # init
    # ------------------------------------------------------------------------
    parser = argparse.ArgumentParser(description="SimCLR")
    config = yaml_config_hook(".\\config\\gan.yaml")
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
    # ------------------------------------------------------------------------
    # Make dir
    # ------------------------------------------------------------------------
    TIME_FN = f"{time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())}-{random.randint(0,10000)}"
    RESULT_FOLDER = join("result", f"{args.fn}_{TIME_FN}")
    os.mkdir(RESULT_FOLDER)
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
            worker(args, X)