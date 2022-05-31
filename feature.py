
import os
import time
from os.path import join
import torch
import pandas as pd
import numpy as np
import torchvision.models as torchvision_models
import argparse

from utils.transforms import Transforms
from utils.identity import Identity
from utils.fs_dataset import FSDataset
import utils.vits as vits
from utils import yaml_config_hook


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

def generate_csv(dataset_path, dataset_name, is_support):
    fn = []
    ln = []
    label_cnt = 0
    if is_support:
        classes = os.listdir(dataset_path)
        for cls in classes:
            cls_path = join(dataset_path, cls)
            images = os.listdir(cls_path)
            for img in images:
                img_path = join(dataset_path, join(cls, img))
                fn.append(img_path)
                ln.append(label_cnt)
            label_cnt += 1
        pd.DataFrame({ "path":fn, "label":ln}).to_csv(join("csv", f"{dataset_name}.csv"), header=True, index=False)
        return label_cnt
    else:
        images = os.listdir(dataset_path)
        for img in images:
            img_path = join(dataset_path, img)
            fn.append(img_path)
            ln.append(label_cnt)
        pd.DataFrame({ "path":fn, "label":ln}).to_csv(join("csv", f"{dataset_name}.csv"), header=True, index=False)
        return label_cnt
   

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real")
    config = yaml_config_hook(".\\config\\feature.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))
    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    

    model_list = [
        # arch       encoder        path
     #    ["mocov3", "vit_base", "checkpoint\\vit-b-300ep.pth.tar"],
        ["mocov3", "vit_small", "checkpoint\\vit-s-300ep.pth.tar"],
        #["mocov3", "resnet50", "checkpoint\\r-50-100ep.pth.tar"],
      #   ["mocov3", "resnet50", "checkpoint\\r-50-300ep.pth.tar"],
        # #["mocov3", "resnet50", "checkpoint\\r-50-1000ep.pth.tar"]
    ]
    # ------------------------------------------------------------------------
    # Loading data
    # ------------------------------------------------------------------------
    begin = time.time()

    dataset_ins = args.dataset_path
    dataset_name = dataset_ins[dataset_ins.rfind('\\')+1:]
    
    n_classes = generate_csv(dataset_path=dataset_ins, dataset_name=dataset_name, is_support=args.supp)
    dataset = FSDataset(
            annotations_file = join("csv", f"{dataset_name}.csv"),
            n_classes= n_classes,
            transform=Transforms(size=args.image_size).test_transform,
            target_transform=None
    )
    dataloader = torch.utils.data.DataLoader(
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
        arch = model_ins[0]
        encoder = model_ins[1]
        model_path = model_ins[2]

        if arch == "mocov3":
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
        mn = f"{arch}-{encoder}"
        model_dict[mn] = model
    end = time.time()
    print(f"=> Loading models done. {end-begin:.2f}s")
    # ------------------------------------------------------------------------
    # Computing feature
    # ------------------------------------------------------------------------
    for model_ins in model_list:
        model = model_dict[f"{model_ins[0]}-{model_ins[1]}"]
        print(f"number of parameters of {model_ins[1]}: {sum(param.numel() for param in model.parameters())}")
        model = model.to(args.device)
        model.eval()
        # ------------------------------------------------------------------------
        # Get feature
        # ------------------------------------------------------------------------
        X, y = inference(dataloader, model, args.device)
        np.save(join("feature", f"{dataset_name}-{model_ins[0]}-{model_ins[1]}.npy"), X)
