import math
import os
from re import S
import time
import argparse
import random
import json
from os.path import join
from turtle import shape
from unittest import result
from matplotlib.font_manager import json_load
import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from yaml import load

from linear_classifier import LinearClassifier
from utils import yaml_config_hook
plt.style.use(['science', 'grid','notebook','no-latex'])

RESULT_FOLDER = ""
SUB_DCIT = {
    "0": 4,
    "1": 1,
    "2": 14,
    "3": 8,
    "4": 0,
    "5": 6,
    "6": 7,
    "7": 7,
    "8": 18,
    "9": 3,
    "10": 3,
    "11": 14,
    "12": 9,
    "13": 18,
    "14": 7,
    "15": 11,
    "16": 3,
    "17": 9,
    "18": 7,
    "19": 11,
    "20": 6,
    "21": 11,
    "22": 5,
    "23": 10,
    "24": 7,
    "25": 6,
    "26": 13,
    "27": 15,
    "28": 3,
    "29": 15,
    "30": 0,
    "31": 11,
    "32": 1,
    "33": 10,
    "34": 12,
    "35": 14,
    "36": 16,
    "37": 9,
    "38": 11,
    "39": 5,
    "40": 5,
    "41": 19,
    "42": 8,
    "43": 8,
    "44": 15,
    "45": 13,
    "46": 14,
    "47": 17,
    "48": 18,
    "49": 10,
    "50": 16,
    "51": 4,
    "52": 17,
    "53": 4,
    "54": 2,
    "55": 0,
    "56": 17,
    "57": 4,
    "58": 18,
    "59": 17,
    "60": 10,
    "61": 3,
    "62": 2,
    "63": 12,
    "64": 12,
    "65": 16,
    "66": 12,
    "67": 1,
    "68": 9,
    "69": 19,
    "70": 2,
    "71": 10,
    "72": 0,
    "73": 1,
    "74": 16,
    "75": 12,
    "76": 9,
    "77": 13,
    "78": 15,
    "79": 13,
    "80": 16,
    "81": 19,
    "82": 2,
    "83": 4,
    "84": 6,
    "85": 19,
    "86": 5,
    "87": 5,
    "88": 8,
    "89": 19,
    "90": 18,
    "91": 1,
    "92": 2,
    "93": 15,
    "94": 6,
    "95": 0,
    "96": 17,
    "97": 8,
    "98": 14,
    "99": 13
}
global_sub_dict = {}
global_urp_dict = {}
upr_way_list = []
sub_way_list = []
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

def test_level(args, loaders, criterion):

    sub_way_list = []
    for k in global_sub_dict:
        sub_way_list.append(k)
    def out_level(l_upr, p1, n_way):
        g_sub_list = global_urp_dict[upr_way_list[l_upr]]
        res = [0.0] * n_way
        for g_sub in g_sub_list:
            res_sub = sub_way_list.index(g_sub)
            p1_sub = g_sub_list.index(g_sub)
            res[res_sub] = p1[0][p1_sub].item()
        return torch.tensor([res])

    loss_epoch = 0
    accuracy_epoch = 0
    acc_super = 0
    acc_super_dict = {}
    for k in global_urp_dict:
        acc_super_dict[k] = [0, 0.0]
    
    classifier =  LinearClassifier(args.n_features, len(upr_way_list))
    classifier.load_state_dict(torch.load(join("checkpoint", f"cls-super.pth")))
    classifier = classifier.to(args.device)
    classifier.eval()
           
    loader = loaders['blank'][1]
    for _, (x, y) in enumerate(loader):
        classifier.zero_grad()
        x = x.to(args.device)
        y = y.to(args.device)
        output0 = classifier(x)
        upr = upr_way_list[output0.argmax(1).item()]
        acc_super += 1.0 if sub_way_list[y.argmax(1).item()] in global_urp_dict[upr] else 0
        if sub_way_list[y.argmax(1).item()] not in global_urp_dict[upr]:
            acc = 0.0
            output = output0
        else:
            sub_cls =  LinearClassifier(args.n_features, len(global_urp_dict[upr]))
            sub_cls.load_state_dict(torch.load(join("checkpoint", f"cls-{upr}.pth")))
            sub_cls = sub_cls.to(args.device)
            sub_cls.eval()
            sub_cls.zero_grad()
            output1 = sub_cls(x)
            output = out_level(output0.argmax(1).item(), output1, args.way).to(args.device)
            acc = (output.argmax(1) == y.argmax(1)).sum().item() / y.size(0)
            acc_super_dict[upr][0] += 1
            acc_super_dict[upr][1] += acc
            loss = criterion(output, y)
        accuracy_epoch += acc
        #loss_epoch += loss.item()
    print(f"[M]acc_super:{acc_super/len(loader)} acc_sub:{accuracy_epoch/acc_super} acc_epoch:{accuracy_epoch/len(loader)}")
    for k in global_urp_dict:
        print(f"\t[{k}]{acc_super_dict[k][1]/(acc_super_dict[k][0] +0.000000001)}")
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
        query, batch_size=(args.logistic_batch_size), shuffle=True
    )

    for way_ins in sample_way:
        global_sub_dict[way_ins] = SUB_DCIT[str(way_ins)]
    
    for k in global_sub_dict:
        urp = global_sub_dict[k]
        if urp not in global_urp_dict:
            global_urp_dict[urp] = [k]
        else:
            global_urp_dict[urp].append(k)
        if urp not in upr_way_list:
            upr_way_list.append(urp)
    
    sub_way_list = sample_way
    print(sub_way_list)
    print(upr_way_list)
    print(global_sub_dict)
    print(global_urp_dict)

    def getLoader(support_X_list, support_y_list, query_X_list, query_y_list,n_label, urp, mfunc):
        assert len(support_X_list) == len(support_y_list)
        assert len(query_X_list) == len(query_y_list)
        def getList(in_x, in_y, n_label, mfunc):
            x = []
            y = []
            for i in range(len(in_x)):
                tmp_label = [0.0] * n_label
                l_sub = in_y[i].index(1.0)
                tmp_label[mfunc(l_sub, urp)] = 1.0
                x.append(in_x[i])
                y.append(tmp_label)
            return [x, y]
        def list2loader(xy):
            return  torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(
                    torch.from_numpy(np.array(xy[0])), torch.from_numpy(np.array(xy[1]))),
                batch_size=args.logistic_batch_size,shuffle=True,)
        return [list2loader(getList(support_X_list, support_y_list, n_label, mfunc)),
                list2loader(getList(query_X_list, query_y_list, n_label, mfunc))]


    def genSuper(n, urp):
        g_sub = sub_way_list[n]
        g_urp = global_sub_dict[g_sub]
        return upr_way_list.index(g_urp)
    
    def genSub(n, urp):
        g_sub = sub_way_list[n]
        if urp != global_sub_dict[g_sub]:
            return -1
        else:
            return global_urp_dict[urp].index(g_sub)

    loaders = {}
    loaders['blank'] = [arr_support_loader, arr_query_loader]
    loaders['super'] = getLoader(support_X_list, support_y_list, query_X_list, query_y_list, len(upr_way_list), -1, mfunc=genSuper)
    for urp in global_urp_dict:
        loaders[urp] = getLoader(support_X_list, support_y_list, query_X_list, query_y_list, len(global_urp_dict[urp]), urp, mfunc=genSub)
    return loaders

def get_mean_support_feature(args, arr_support_loader, way):
    mean_support_feature = torch.zeros(way, args.n_features)
    for step, (x, y) in enumerate(arr_support_loader):
        mean_support_feature[y.argmax()] += x[0]
    mean_support_feature = F.normalize(mean_support_feature / args.shot)
    return mean_support_feature

def worker(args, X):
    begin = time.time()
    sample_acc = 0.0
    for epi in range(args.sample_epoch):
        
        loaders = sample_from_loader(X, args)
        criterion = torch.nn.CrossEntropyLoss()
        
        for k in loaders:
            if k == 'blank':
                way = args.way
            elif k == 'super':
                way = len(upr_way_list)
            else:
                way = len(global_urp_dict[k])
            support_loader = loaders[k][0]
            mean_support_feature = get_mean_support_feature(args, support_loader, way) \
                if args.support_init_enabled else None
            classifier = LinearClassifier(args.n_features, way, weight=mean_support_feature)
            classifier = classifier.to(args.device)
            optimizer = torch.optim.Adam(classifier.parameters(), lr=3e-4)
            if type(k) == type(1) and len(global_urp_dict[k]) < 2:
                print(f"[{k}](test) loss:0.0 acc:1.0")
            else:
                for epoch in range(args.logistic_epochs):
                    loss_epoch, accuracy_epoch = train(args, loaders[k][0], classifier, criterion, optimizer)
                print(f"[{k}](train) loss:{loss_epoch} acc:{accuracy_epoch}")
                loss_epoch, accuracy_epoch = test(args, loaders[k][1], classifier, criterion)
                print(f"[{k}](test) loss:{loss_epoch} acc:{accuracy_epoch}")
            torch.save(classifier.state_dict(), join("checkpoint", f"cls-{k}.pth"))

        
        loss_epoch, accuracy_epoch = test_level(args, loaders , criterion)
        print(f"[M](support) loss:{loss_epoch} acc:{accuracy_epoch}")
        loss_epoch, accuracy_epoch = test_level(args, loaders, criterion)
        print(f"[M](query) loss:{loss_epoch} acc:{accuracy_epoch}")

            

    sample_acc /= args.sample_epoch
    print(f"{args.dataset} {args.arch} {args.encoder} {args.way}way {args.shot}shot {sample_acc:.6f} {time.time()-begin:.2f}s")
    return sample_acc

if __name__ == "__main__" :
    # ------------------------------------------------------------------------
    # init
    # ------------------------------------------------------------------------
    parser = argparse.ArgumentParser(description="SimCLR")
    config = yaml_config_hook(".\\config\\multi.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))
    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # ------------------------------------------------------------------------
    # Setting
    # ------------------------------------------------------------------------
    dataset_list = [
       #"mini-imagenet",
        "FC100",
        # "tiered_imagenet",
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
        5: [1],#[1, 5, 10, 20],
        10: [1],#, 5, 10, 20],
        20:[1],
        50: [1, 5, 10, 20, 50],
        100: [1, 5, 10, 20, 50],
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