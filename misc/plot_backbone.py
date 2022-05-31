# support init or not
import numpy as np
import matplotlib.pyplot as plt
import json
import math
from os.path import join

plt.style.use([ 'grid','notebook',])
plt.rc('text', usetex=True)

def read_json(fn):
    with open(fn, 'r') as f:
        return json.loads(f.read())


dataset_list = ["mini-imagenet", "FC100",]
ARCH =  [ "vit_base", "vit_small",  "resnet50"]
FOLDERS= ["BASE", "SMALL", "MOCORES50"]
way_list = [2, 5, 10, 50]
shot_list = [1, 5, 10, 20]
way_shot_dict ={
        # way [shot0, shot1, ...]
        2: [1, 5, 10, 20],
        5: [1, 5, 10, 20],
        10: [1, 5, 10, 20],
       50: [1, 5, 10, 20],
    }
BACKBONE_FLODER = 'result\BACKBONE'
def plot_axes(all_res, dataset, way, shot, ax, xla):  
    for i in range(len(FOLDERS)):
        if xla == 'epoch':
            x = all_res[ f"{dataset}-{way}-{shot}-{FOLDERS[i]}"]["plot_x"]
            y = all_res[ f"{dataset}-{way}-{shot}-{FOLDERS[i]}"]["plot_loss_y"]
            test_acc = all_res[ f"{dataset}-{way}-{shot}-{FOLDERS[i]}"][f"test_acc"]
            label = f"{ARCH[i]} (test acc={test_acc:.4f})"
        elif xla == 'shot':
            x = shot_list
            y = []
            for shot in shot_list:
                y.append(all_res[f"{dataset}-{way}-{shot}-{FOLDERS[i]}"]['test_acc'])
            label = f"{ARCH[i]}"
        elif xla == 'way':
            x = way_list[:-1]
            y_e = []
            y = []
            accum = 0
            for way in way_list:
                acc = all_res[f"{dataset}-{way}-{shot}-{FOLDERS[i]}"]['test_acc']
                acc2 = math.pow(acc, 1.0/math.log2(way))
                accum += acc2
                if way == 50:
                    continue
                y_e.append(acc2)
                y.append(acc)
            print(f"{ARCH[i]} {dataset} {shot} {accum*100/len(way_list):.2f}")
            label = f"{ARCH[i]}"
            ax.plot(np.array(x), np.array(y_e), '^-', color=f"C{i}", label=f"{label} (2)")

        ax.plot(np.array(x), np.array(y), 'o-', color=f"C{i}", label=label)
   # ax.set_title(f'')
    ax.set_xlabel(xla)
    if xla == 'epoch':
        ax.set_title(f'{way}way, {shot}shot ({dataset})')
        ax.set_ylabel('loss')
    elif xla == 'shot':
        ax.set_title(f'{way}-way, n-shot ({dataset})')
        ax.set_ylabel('acc')
        ax.set_ylim(0.0, 1.0)
    elif xla == 'way':
        ax.set_title(f'm-way, {shot}-shot ({dataset})')
        ax.set_ylabel('acc')
    ax.legend()

def plot_axes_way(all_res, dataset, shot, ax, xla):
    WAY = 5
    for i in range(len(FOLDERS)):
        if xla == 'epoch':
            x = all_res[ f"{dataset}-{WAY}-{shot}-{FOLDERS[i]}"]["plot_x"]
            y = all_res[ f"{dataset}-{WAY}-{shot}-{FOLDERS[i]}"]["plot_loss_y"]
            test_acc = all_res[ f"{dataset}-{WAY}-{shot}-{FOLDERS[i]}"][f"test_acc"]
            label = f"{ARCH[i]} (test acc={test_acc:.4f})"
        elif xla == 'shot':
            x = shot_list
            y = []
            for shot in shot_list:
                y.append(all_res[f"{dataset}-{WAY}-{shot}-{FOLDERS[i]}"]['test_acc'])
            label = f"{ARCH[i]}"
        elif xla == 'way':
            x = way_list[:-1]
            y_e = []
            y = []
            accum = 0
            for way in way_list[:-1]:
                acc = all_res[f"{dataset}-{way}-{shot}-{FOLDERS[i]}"]['test_acc']
                acc2 = math.pow(acc, 1.0/math.log2(way))
                y_e.append(acc2)
                y.append(acc)
                accum += acc2
            print(f"{ARCH[i]} {accum/len(way_list)}")
            label = f"{ARCH[i]}"
            ax.plot(np.array(x), np.array(y_e), '^-', color=f"C{i}", label=f"{label} (2)")
        ax.plot(np.array(x), np.array(y), 'o-', color=f"C{i}", label=label)
    ax.set_title(f'{dataset}')
    ax.set_xlabel(xla)
    if xla == 'epoch':
        ax.set_ylabel('loss')
    elif xla == 'shot':
        ax.set_ylabel('acc')
        ax.set_ylim(0.0, 1.0)
    elif xla == 'way':
        ax.set_ylabel('acc')
    ax.legend()

def avg_dict(fn_prefix, epi):
    res_dict = {}
    for i in range(epi):
        if len(res_dict) == 0:
            res_dict = read_json(f'{fn_prefix}-{i}.json')
        else:
            tmp_dict = read_json( f'{fn_prefix}-{i}.json')
            for k in res_dict:
                if type(res_dict[k]) == type([]):
                    for j in range(len(res_dict[k])):
                        res_dict[k][j] += tmp_dict[k][j]
                else:
                    res_dict[k] += tmp_dict[k]
    for k in res_dict:
        if type(res_dict[k]) == type([]):
            for j in range(len(res_dict[k])):
                res_dict[k][j] /= 10
        else:
            res_dict[k] /= 10  
    return res_dict

all_res = {}
for dataset_ins in dataset_list:
    for way in way_list:
        for floder in FOLDERS:
            for shot in way_shot_dict[way]:
                path = join(f"{BACKBONE_FLODER}_{floder}_{way}", f"{dataset_ins}-{way}way-{shot}shot")
                k = f"{dataset_ins}-{way}-{shot}-{floder}"
                all_res[k] = avg_dict(fn_prefix=path, epi=(2 if way == 50 else 10))


for dataset in dataset_list:
    fig, axs = plt.subplots(nrows=len(way_list), ncols=len(shot_list), figsize=(6*(len(shot_list)), 6*(len(way_list))))
    for i in range(len(way_list)):
        for j in range(len(shot_list)):
            plot_axes(all_res,dataset, way_list[i], shot_list[j], axs[i, j], 'epoch')
    fig.savefig(f'appendix-{dataset}-loss.png')
    fig, axs = plt.subplots(nrows=2, ncols=4, figsize=(6*4, 6*2))
    for i in range(len(way_list)):
        plot_axes(all_res, dataset, way_list[i], shot_list[0], axs[0,i], 'shot')
    for j in range(len(shot_list)):
        plot_axes(all_res, dataset, way_list[0], shot_list[j], axs[1, j], 'way')
    fig.savefig(f'appendix-{dataset}-ws.png')
    #.text(4, 1, "QWBDUWBDUBDUWB", ha='left', wrap=True)

fig, axs = plt.subplots(nrows=len(dataset_list), ncols=1+2, figsize=(8*(1+2), 8*len(dataset_list)))
for i in range(len(dataset_list)):
    plot_axes_way(all_res, dataset_list[i],5, axs[i, 0], 'epoch')
    plot_axes_way(all_res,  dataset_list[i], 5, axs[i, 1], 'shot')
    plot_axes_way(all_res,  dataset_list[i], 5, axs[i, 2], 'way')

fig.savefig('backbone-5way-5shot-.png')
