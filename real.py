import math
import os
import time
import argparse
import csv
import json
from os.path import join
import torch
import numpy as np
import torch.nn.functional as F

from utils.linear_classifier import LinearClassifier
from utils import yaml_config_hook

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
    
def move_img(args, dm):
    img_list = []
    with open(args.query_csv,'r') as f:
        reader = csv.reader(f)
        img_list = [row[0] for row in reader]

    img_list = img_list[1:] # remove header of csv
    fn_list = [path[path.rfind('\\') + 1:] for path in img_list]
    print(img_list[0])
    print(fn_list[0])

    out_path = f"{args.out_path}-{time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())}"

    os.makedirs(out_path)
    for i in range(args.way):
        subfolder = join(out_path, f"{i}")
        os.makedirs(subfolder)
    
    for i in range(len(img_list)):
        path = img_list[i]
        dst = join(join(out_path, f"{dm[i]}"), fn_list[i])
        with open(path, 'rb') as f:
            with open(dst, 'wb') as d:
                d.write(f.read())

def pred(args, loader, classifier):

    classifier.eval()
    dict_img = {}
    for step, (x, y) in enumerate(loader):
        classifier.zero_grad()

        x = x.to(args.device)
        
        output = classifier(x)

        label = output.argmax(1)
        dict_img[step] = label.item()
    return dict_img
      
def sample_from_loader(X, args):

    X_list = []
    y_list = []
    for way_ins in range(args.way):
        tmp_label = [0.0] * args.way
        tmp_label[way_ins] = 1.0
        for shot_ins in range(args.shot):
            X_list.append(X[way_ins * args.shot + shot_ins])
            y_list.append(tmp_label)

    dataset = torch.utils.data.TensorDataset(torch.from_numpy(np.array(X_list)), torch.from_numpy(np.array(y_list)))
    arr_loader = torch.utils.data.DataLoader(dataset, batch_size=args.logistic_batch_size, shuffle=True,)
    return arr_loader

def sample_from_loaderQ(X, args):

    X_list = []
    y_list = []
    for i in range(len(X)):
        X_list.append(X[i])
        y_list.append([0.0] * args.way)

    dataset = torch.utils.data.TensorDataset(torch.from_numpy(np.array(X_list)), torch.from_numpy(np.array(y_list)))
    arr_loader = torch.utils.data.DataLoader(dataset, batch_size=args.logistic_batch_size, shuffle=False,)
    return arr_loader

def get_mean_support_feature(args, arr_support_loader):
    mean_support_feature = torch.zeros(args.way, args.n_features)
    for step, (x, y) in enumerate(arr_support_loader):
        mean_support_feature[y.argmax()] += x[0]
    mean_support_feature = F.normalize(mean_support_feature / args.shot)
    return mean_support_feature

if __name__ == "__main__" :
    # ------------------------------------------------------------------------
    # init
    # ------------------------------------------------------------------------
    parser = argparse.ArgumentParser(description="Real")
    config = yaml_config_hook(".\\config\\real.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))
    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # ------------------------------------------------------------------------
    # Setting
    # ------------------------------------------------------------------------
    model_list = [
        # arch       encoder      
        # ["mocov3", "vit_base"],
        ["mocov3", "vit_small",],
        # #["mocov3", "resnet50"],
    ]

    
    for model_ins in model_list:
        args.arch = model_ins[0]
        args.encoder = model_ins[1]
        SX = np.load(args.support_npy)
        QX = np.load(args.query_npy)
        args.n_features = SX.shape[1]
        arr_support_loader = sample_from_loader(SX, args)
        arr_query_loader = sample_from_loaderQ(QX, args)

        classifier = LinearClassifier(args.n_features, args.way, weight=get_mean_support_feature(args, arr_support_loader))
        classifier = classifier.to(args.device)
        optimizer = torch.optim.Adam(classifier.parameters(), lr=3e-4)
        criterion = torch.nn.CrossEntropyLoss()

        for epoch in range(args.logistic_epochs):
            loss_epoch, accuracy_epoch = train(args, arr_support_loader, classifier, criterion, optimizer)
            if epoch % 10 == 0:
                print(f"[{epoch}/{args.logistic_epochs}] loss={loss_epoch} acc={accuracy_epoch}")
            if args.mixup_enabled:
                mixup(args, arr_support_loader, classifier, criterion, optimizer)
        dict_img = pred(args, arr_query_loader, classifier)
        move_img(args, dict_img)
