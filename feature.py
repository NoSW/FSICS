
import os
import time
from os.path import join
import torch
import torchvision
import pandas as pd
import numpy as np
import torchvision.models as torchvision_models

from simclr.modules.transformations import TransformsSimCLR
from simclr.modules.identity import Identity
from fs_dataset import FSDataset
import vits


def inference(loader, model, device):
    begin = time.time()
    feature_vector = []
    labels_vector = []
    for step, (x, y) in enumerate(loader):
        x = x.to(device)
        with torch.no_grad():
            h = model(x)

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

def generate_csv(dataset, subfolders = ["train", "test", "val"]):
    dataset_path = join("dataset", dataset)
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
    pd.DataFrame({ "path":fn, "label":ln}).to_csv(join("csv", f"{dataset}.csv"), header=True, index=False)
    return label_cnt

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_size = 224
    batch_size = 128
    workers = 8

    dataset_list = [
       # "mini-imagenet",
        "FC100",
    ]
    model_list = [
        # arch       encoder        path
        # ["mocov3", "vit_base", "checkpoint\\vit-b-300ep.pth.tar"],
      #  ["mocov3", "vit_small", "checkpoint\\vit-s-300ep.pth.tar"],
        #["mocov3", "resnet50", "checkpoint\\r-50-100ep.pth.tar"],
         ["mocov3", "resnet50", "checkpoint\\r-50-300ep.pth.tar"],
        # #["mocov3", "resnet50", "checkpoint\\r-50-1000ep.pth.tar"],
        # ["resnet", "resnet18",""],
        # ["resnet", "resnet50",""],
    ]
    # ------------------------------------------------------------------------
    # Loading data
    # ------------------------------------------------------------------------
    begin = time.time()
    dataset_dict = {}
    for dataset_ins in dataset_list:
    
        n_classes = generate_csv(dataset=dataset_ins , subfolders=['all'])
        #
        dataset = FSDataset(
                annotations_file = join("csv", f"{dataset_ins}.csv"),
                n_classes= n_classes,
                transform=TransformsSimCLR(size=image_size).test_transform,
                target_transform=None
        )
        dataset_dict[dataset_ins] = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size,
            shuffle=False, drop_last=False, num_workers=workers)

    end = time.time()
    print(f"=> Loading datasets done. {end-begin:.2f}s ")

    # ------------------------------------------------------------------------
    # Loading model
    # ------------------------------------------------------------------------
    begin = time.time()
    model_dict = {}
    for model_ins in model_list:
        arch = model_ins[0]
        encoder = model_ins[1]
        model_path = model_ins[2]

        if arch == "resnet":
            if encoder == "resnet18":
                model =  torchvision.models.resnet18(pretrained=True)
            elif encoder == "resnet50":
                model =  torchvision.models.resnet50(pretrained=True)
        elif arch == "mocov3":
            if encoder.startswith('vit'):
                model = vits.__dict__[encoder]()
                linear_keyword = 'head'
            elif encoder == "resnet50":
                model = torchvision_models.__dict__[encoder]()
                linear_keyword = 'fc'
            else:
                raise NotImplementedError
            checkpoint = torch.load(model_path, map_location="cpu")
            state_dict = checkpoint['state_dict']
            for k in list(state_dict.keys()):
                # retain only base_encoder up to before the embedding layer
                if k.startswith('module.base_encoder') and not k.startswith('module.base_encoder.%s' % linear_keyword):
                    # remove prefix
                    state_dict[k[len("module.base_encoder."):]] = state_dict[k]
                # delete renamed or unused k
                del state_dict[k]
            model.load_state_dict(state_dict, strict=False)
        else:
            raise NotImplementedError
        if encoder.startswith('vit'):
            n_features = model.head.in_features
            model.head = Identity()
            image_size = 224
        else:
            n_features = model.fc.in_features  # get dimensions of fc layer
            model.fc = Identity()
        # load pre-trained model from checkpoint
        mn = f"{arch}-{encoder}"
        model_dict[mn] = model
    end = time.time()
    print(f"=> Loading models done. {end-begin:.2f}s")

    # ------------------------------------------------------------------------
    # Computing feature
    # ------------------------------------------------------------------------
    for dataset_ins in dataset_list:
        dataloader = dataset_dict[dataset_ins]
        for model_ins in model_list:
            model = model_dict[f"{model_ins[0]}-{model_ins[1]}"]
            model = model.to(device)
            model.eval()
            # ------------------------------------------------------------------------
            # Get feature
            # ------------------------------------------------------------------------
            X, y = inference(dataloader, model, device)
            np.save(join("feature", f"{dataset_ins}-{model_ins[0]}-{model_ins[1]}.npy"), X)
    