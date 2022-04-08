import os
from os.path import join
import pandas as pd

DATASET = "dataset\\mini-imagenet"
SUBFOLDER = "train"
PATH = join(DATASET, SUBFOLDER)
SUPPORT_SET = join(DATASET, "support_set.csv")
QUERY_SET = join(DATASET, "query_set.csv")

WAY = 100
SUPPORT_SHOT = 600
QUERY_SHOT = 0

classes = os.listdir(PATH)
support_fn = []
support_ln = []
query_fn = []
query_ln = []

label_cnt = 0

def read_images(fn, ln):
    global label_cnt
    for cls in classes:
        cls_path = join(PATH, cls)
        images = os.listdir(cls_path)
        for img in images:
            fn.append(join(SUBFOLDER, join(cls, img)))
            ln.append(label_cnt)
        label_cnt += 1

if __name__ == "__main__":

    

    read_images(support_fn, support_ln)
    SUBFOLDER = 'test'
    PATH = join(DATASET, SUBFOLDER)
    classes = os.listdir(PATH)
    read_images(support_fn, support_ln)
    SUBFOLDER = 'val'
    PATH = join(DATASET, SUBFOLDER)
    classes = os.listdir(PATH)
    read_images(support_fn, support_ln)



    pd.DataFrame({ "path":support_fn, "label":support_ln}).to_csv(SUPPORT_SET, header=True, index=False)
    # pd.DataFrame({ "path":query_fn, "label":query_ln}).to_csv(QUERY_SET, header=True, index=False)
