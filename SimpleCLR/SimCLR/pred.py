import math
import os
import argparse
import torch
import torchvision

import numpy as np

from simclr import SimCLR

from linear_classifier import LinearClassifier
from simclr.modules.transformations import TransformsSimCLR

from utils import yaml_config_hook
from torch.utils.data import DataLoader
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
    feature_vector = []
    labels_vector = []
    for step, (x, y) in enumerate(loader):
        x = x.to(device)

        # get encoding
        with torch.no_grad():
            h, _, z, _ = model(x, x)

        h = h.detach()

        feature_vector.extend(h.cpu().detach().numpy())
        labels_vector.extend(y.numpy())

        if step % 20 == 0:
            print(f"Step [{step}/{len(loader)}]\t Computing features...")

    feature_vector = np.array(feature_vector)
    labels_vector = np.array(labels_vector)
    print("Features shape {}".format(feature_vector.shape))
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SimCLR")
    config = yaml_config_hook("./config/pred.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))

    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    

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

    print(encoder)

    if args.encoder.startswith('vit'):
        n_features = encoder.head.in_features
        encoder.head = Identity()
        args.image_size = 224
    else:
        n_features = encoder.fc.in_features  # get dimensions of fc layer
        encoder.fc = Identity()

    # load pre-trained model from checkpoint
    model = SimCLR(encoder, args.projection_dim, n_features)
    if args.arch == 'simclr':
        model.load_state_dict(torch.load(args.model_path, map_location=args.device.type))
    model = model.to(args.device)
    model.eval()

    print(model)
    # -------------------------------------------------


    train_dataset = FSDataset(
            annotations_file ="../../dataset/mini-imagenet/support_set.csv",
            img_dir = "..\..\dataset\mini-imagenet",
            n_classes= args.n_classes,
            transform=TransformsSimCLR(size=args.image_size).test_transform,
            target_transform=None
    )
    test_dataset = FSDataset(
            annotations_file = "../../dataset/mini-imagenet/query_set.csv",
            img_dir = "..\..\dataset\mini-imagenet",
            n_classes= args.n_classes,
            transform=TransformsSimCLR(size=args.image_size).test_transform,
            target_transform=None
    )
    

    support_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.logistic_batch_size,
        shuffle=True, drop_last=False, num_workers=args.workers)

    query_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size= args.logistic_batch_size,
        shuffle=False, drop_last=False, num_workers=args.workers)

    # -------------------------------------------------

    print("### Creating features from pre-trained context model ###")
    support_X, support_y = inference(support_loader, model, args.device)
    query_X, query_y = inference(query_loader, model,  args.device)
   
    arr_support_loader, arr_query_loader = create_data_loaders_from_arrays(
        support_X, support_y, query_X, query_y, args.logistic_batch_size
    )

    mean_support_feature = torch.zeros(args.n_classes, model.n_features)
    for step, (x, y) in enumerate(arr_support_loader):
        mean_support_feature[y.argmax()] += x[0]
    n_shot = len(arr_support_loader) / args.n_classes
    mean_support_feature = F.normalize(mean_support_feature / n_shot)

    classifier = LinearClassifier(model.n_features, args.n_classes,
        weight=mean_support_feature, bias=torch.zeros(args.n_classes))

    classifier = classifier.to(args.device)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=3e-4)
    criterion = torch.nn.CrossEntropyLoss()

    # fine-tuning
    for epoch in range(args.logistic_epochs):
        # loss_epoch, accuracy_epoch = regular(args, arr_query_loader, classifier,  optimizer)
        # print(
        #     f"Epoch [{epoch}/{args.logistic_epochs}]\t Loss: {loss_epoch / len(arr_query_loader)}\t Accuracy: {accuracy_epoch / len(arr_query_loader)}"
        # )
        loss_epoch, accuracy_epoch = train(
            args, arr_support_loader, classifier, criterion, optimizer
        )
        print(
            f"Epoch [{epoch}/{args.logistic_epochs}]\t Loss: {loss_epoch / len(arr_support_loader)}\t Accuracy: {accuracy_epoch / len(arr_support_loader)}"
        )

    # # entropy_regularization
    # for epoch in range(args.logistic_epochs):
    #     loss_epoch, accuracy_epoch = regular(args, arr_query_loader, classifier,  optimizer)
    #     print(
    #         f"Epoch [{epoch}/{args.logistic_epochs}]\t Loss: {loss_epoch / len(arr_query_loader)}\t Accuracy: {accuracy_epoch / len(arr_query_loader)}"
    #     )

    # final testing
    loss_epoch, accuracy_epoch = test(
        args, arr_query_loader, classifier, criterion, optimizer
    )
    print(
        f"[FINAL]\t Loss: {loss_epoch / len(arr_query_loader)}\t Accuracy: {accuracy_epoch / len(arr_query_loader)}"
    )


  