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
import pandas as pd
from linear_classifier import LinearClassifier
from utils import yaml_config_hook
from scipy import stats

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

def sample_from_loader(X, args, sample_way):    
    support_X_list = []
    support_y_list = []
    query_X_list = []
    query_y_list = []
    for way_ins in sample_way:
        tmp_label = [0.0] * args.way
        tmp_label[sample_way.index(way_ins)] = 1.0
        for shot_ins in range(args.sample_per_class):
            v = X[way_ins * args.sample_per_class + shot_ins]
            if shot_ins >= args.shot:
                query_X_list.append(v)
                query_y_list.append(tmp_label)
            else:
                support_X_list.append(v)
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

def get_stat_dict(X, args, sample_way, n_shot, dataset='FC100'):
    stat_dict = {}
    for way_ins in sample_way:
        tmp = []
        for shot_ins in range(n_shot):
            v = X[way_ins * args.sample_per_class + shot_ins]
            tmp.append(v)
        df = pd.DataFrame(np.array(tmp))
        stat_dict[way_ins] = {
            'sample_way': sample_way,
            'n_sample': n_shot,
            'way': way_ins,
            'Ex':df.mean().to_list(),
            'Vx': df.std().to_list(),
            'Skew': df.skew().to_list() if n_shot >= 8 else 'NAN',
            'Kur': df.kurtosis().to_list()  if n_shot >= 8 else 'NAN',
            'norm_test':list(stats.normaltest(tmp))  if n_shot >= 8 else 'NAN',
           # 'ad':stats.anderson(tmp),
            'ori': tmp,
        }
        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                return json.JSONEncoder.default(self, obj)
        with open(f'mapping\\mapping_5_9875_6467_{way_ins}way_{dataset}.json', 'w') as f:
            json.dump(stat_dict[way_ins], f, cls=NumpyEncoder)
    
    stat_dict['Ex_avg'] = [0.0] * 384
    stat_dict['Vx_avg'] = [0.0] * 384
    for d in range(384):
        for way_ins in sample_way:
            stat_dict['Ex_avg'][d] += stat_dict[way_ins]['Ex'][d]
            stat_dict['Vx_avg'][d] += stat_dict[way_ins]['Vx'][d]
        stat_dict['Ex_avg'][d] /= len(sample_way)
        stat_dict['Vx_avg'][d] /= len(sample_way) * len(sample_way)
    return stat_dict

def sample_from_loader_mapping(XS, XD, args, sample_way, stat_dict_S, stat_way):

    stat_dict_D = get_stat_dict(X=XD, args=args, sample_way=sample_way, n_shot= args.shot)

    wm = {}
    for i in range(len(sample_way)):
        wm[sample_way[i]] = stat_way[i]
    
    support_X_list = []
    support_y_list = []
    query_X_list = []
    query_y_list = []
    for way_ins in sample_way:
        tmp_label = [0.0] * args.way
        tmp_label[sample_way.index(way_ins)] = 1.0
        for shot_ins in range(args.shot):
            v = XD[way_ins * args.sample_per_class + shot_ins]
            support_X_list.append(v)
            support_y_list.append(tmp_label)
        for shot_ins in range(min(args.shot * 100, 600)):
            way_src = wm[way_ins]
            v = XS[way_src * args.sample_per_class + shot_ins]
            for d in range(len(v)):
                v[d] = (((v[d] - stat_dict_S[way_src]['Ex'][d])/stat_dict_S[way_src]['Vx'][d])*\
                    stat_dict_D[way_ins]['Vx'][d]) + stat_dict_D[way_ins]['Ex'][d]
            support_X_list.append(v)
            support_y_list.append(tmp_label)


    for way_ins in sample_way:
        tmp_label = [0.0] * args.way
        tmp_label[sample_way.index(way_ins)] = 1.0
        for shot_ins in range(args.shot, args.sample_per_class):
            v = XD[way_ins * args.sample_per_class + shot_ins]
            query_X_list.append(v)
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
        query, batch_size=(args.logistic_batch_size * 8), shuffle=True
    )
    return arr_support_loader, arr_query_loader

def get_mean_support_feature(args, arr_support_loader):
    mean_support_feature = torch.zeros(args.way, args.n_features)
    for step, (x, y) in enumerate(arr_support_loader):
        mean_support_feature[y.argmax()] += x[0]
    mean_support_feature = F.normalize(mean_support_feature / (len(arr_support_loader)/args.way))
    return mean_support_feature

def instance(args, arr_support_loader, arr_query_loader, trained):
    mean_support_feature = get_mean_support_feature(args, arr_support_loader) \
            if args.support_init_enabled else None
    classifier = LinearClassifier(args.n_features, args.way, weight=mean_support_feature)
    classifier = classifier.to(args.device)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=3e-4)
    criterion = torch.nn.CrossEntropyLoss()
    if trained:
        for epoch in range(args.logistic_epochs):
            loss_epoch, accuracy_epoch = train(args, arr_support_loader, classifier, criterion, optimizer)
    loss_epoch, accuracy_epoch = test(args, arr_query_loader, classifier, criterion, optimizer)
    
    return accuracy_epoch

def worker(args, FC, Mini):
    begin = time.time()
    sample_acc = {'fc': 0.0, 'fc(mini)_max': 0.0, 'fc(mini)_min':0.0,
    'fc_no': 0.0, 'fc(mini)_max_no': 0.0, 'fc(mini)_min_no':0.0,
    }
    sample_way_list = [random.sample(range(0, args.n_classes), args.way) for i in range(args.sample_epoch)]
    for epi in range(args.sample_epoch):
        sample_way = sample_way_list[epi]
        arr_support_loader_fc, arr_query_loader_fc = sample_from_loader(X=FC, args=args, sample_way=sample_way)
        acc_no = 0 #instance(args, arr_support_loader_fc, arr_query_loader_fc, trained=False)
        acc = instance(args, arr_support_loader_fc, arr_query_loader_fc, trained=True)
        sample_acc["fc"] += acc
        sample_acc["fc_no"] += acc_no
        print(f"fc acc:{acc} acc_no:{acc_no} {sample_way}")

    
    stat_way_best = [56, 41, 99, 12, 58]
    stat_dict_S_best = get_stat_dict(X=Mini, args=args, sample_way=stat_way_best, n_shot= args.sample_per_class, dataset='mini=imagenet')
    for epi in range(args.sample_epoch):
        sample_way = sample_way_list[epi]
        arr_support_loader_df, arr_query_loader_df = sample_from_loader_mapping(
           XS=Mini, XD=FC, args=args, sample_way=sample_way, stat_dict_S=stat_dict_S_best ,stat_way=stat_way_best)
        acc_no = 0# instance(args, arr_support_loader_df, arr_query_loader_df, trained=False)
        acc = instance(args, arr_support_loader_df, arr_query_loader_df, trained=True)
        sample_acc['fc(mini)_max'] += acc
        sample_acc['fc(mini)_max_no'] += acc_no
        print(f"fc(mini)_max acc:{acc} acc_no:{acc_no} {sample_way}")

    stat_way_worst = [46, 43, 22, 35, 36]
    stat_dict_S_worst = get_stat_dict(X=Mini, args=args, sample_way=stat_way_worst, n_shot= args.sample_per_class, dataset='mini=imagenet')
    for epi in range(args.sample_epoch):
        sample_way = sample_way_list[epi]
        arr_support_loader_df, arr_query_loader_df = sample_from_loader_mapping(
           XS=Mini, XD=FC, args=args, sample_way=sample_way, stat_dict_S=stat_dict_S_worst ,stat_way=stat_way_worst)
        acc_no = 0#instance(args, arr_support_loader_df, arr_query_loader_df, trained=False)
        acc = instance(args, arr_support_loader_df, arr_query_loader_df, trained=True)
        sample_acc['fc(mini)_min'] += acc
        sample_acc['fc(mini)_min_no'] += acc_no
        print(f"fc(mini)_min acc:{acc} acc_no:{acc_no} {sample_way}")
    
    for k in sample_acc:
        sample_acc[k] /= args.sample_epoch
        print(f"{k} {args.arch} {args.encoder} {args.way}way {args.shot}shot {sample_acc[k]:.6f} {time.time()-begin:.2f}s")

if __name__ == "__main__" :
    # ------------------------------------------------------------------------
    # init
    # ------------------------------------------------------------------------
    parser = argparse.ArgumentParser(description="support based initialization")
    config = yaml_config_hook(".\\config\\mapping.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))
    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # ------------------------------------------------------------------------
    # Setting
    # ------------------------------------------------------------------------
    model_list = [
        # arch       encoder      
      #  ["mocov3", "vit_base"],
        ["mocov3", "vit_small",],
       # ["mocov3", "resnet50"],
    ]
    way_shot_dict ={
        # way [shot0, shot1, ...]
        2: [2, 5, 10, 20],
        5: [2,5,10, 20],
        10:[2, 5, 10, 20],
        50: [2, 5, 10, 20],
    }

    # ------------------------------------------------------------------------
    # Run DxExN iters
    # ------------------------------------------------------------------------
    args.dataset = 'FC100'
    for model_ins in model_list:
        args.arch = model_ins[0]
        args.encoder = model_ins[1]
        # ------------------------------------------------------------------------
        # Get feature
        # ------------------------------------------------------------------------
        X_FC = np.load(join("feature", f"FC100-{args.arch}-{args.encoder}.npy"))
        X_MINI = np.load(join("feature", f"mini-imagenet-{args.arch}-{args.encoder}.npy"))
        args.n_features = X_FC.shape[1]
        # ------------------------------------------------------------------------
        # M-way, N-shot
        # ------------------------------------------------------------------------
        for shot_ins in way_shot_dict[args.way]:
            args.shot = shot_ins
            worker(args=args, FC=X_FC, Mini=X_MINI)