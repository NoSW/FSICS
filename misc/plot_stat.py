from os.path import join
import matplotlib.pyplot as plt
import numpy as np
import math
import json
import os
SHOT = 600
WAY = 100

dataset_list = [
        "mini-imagenet",
        "FC100",
    ]
model_list = [
    # arch       encoder      
    ["mocov3", "vit_base"],
    ["mocov3", "vit_small",],
    ["mocov3", "resnet50"],
]

def read_json(fn):
    with open(fn, 'r') as f:
        return json.loads(f.read())

plot_dict = read_json('stat_plot.json')

fig, ax =  plt.subplots(nrows=len(dataset_list), ncols=2, figsize=(10*2, 10*len(dataset_list)))
for i in range(len(dataset_list)):
    for j in range(len(model_list)):
        var_list = plot_dict[f"{dataset_list[i]}-{model_list[j][1]}-var"]
        ax[i, 0].plot(np.array(range(len(var_list))), np.array(var_list), 'o-', color=f"C{j}", label=f"{model_list[j][1]}")
        ax[i, 0].set_xlabel('class_id')
        ax[i, 0].set_ylabel('distance')
        ax[i, 0].set_title(f'Average distance within the class ({dataset_list[i]})')
        ax[i, 0].set_ylim(0, 0.004)
        ax[i, 0].legend()

        dis_list = plot_dict[f"{dataset_list[i]}-{model_list[j][1]}-dis"]
        dis_list.sort()
        ax[i, 1].plot(np.array( range(len(dis_list))), np.array(dis_list), 'o-', color=f"C{j}", label=f"{model_list[j][1]}")
        ax[i, 1].set_xlabel('class_pair_id')
        ax[i, 1].set_ylabel('distance')
        ax[i, 1].set_ylim(0.0, 0.0016)
        ax[i, 1].set_title(f'Average distance between classes({dataset_list[i]})')
        ax[i, 1].legend()
fig.savefig('stat.png')