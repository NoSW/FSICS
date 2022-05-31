from os.path import join
import matplotlib.pyplot as plt
import numpy as np
import math
import json
import random

SHOT = 600
WAY = 100

shot_lists = [
    random.sample(range(0, 768), 384),
    random.sample(range(0, 384), 384),
    random.sample(range(0, 2048), 384),
]    


def get_squre(a, b, dim=1):
    if dim == 1:
        return abs(a - b)
    else:
        res = 0
        for d in range(dim):
            res +=(a[d]-b[d]) * (a[d]-b[d])
        return res

def get_distance(a, b, dim=1):
    return math.sqrt(get_squre(a, b, dim=dim))

def get_median(a):
    tmp = []
    for v in a:
        tmp.append(v)
    tmp.sort()
    return tmp[len(tmp)//2]

if __name__ == "__main__" :
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
    res_dict = {}
    plot_dict = {}
    for dataset_ins in dataset_list:
        for model_ins in model_list:
            X = np.load(join("feature", f"{dataset_ins}-{model_ins[0]}-{model_ins[1]}.npy"))
            center = []
            print(X.shape)
            n_features = X.shape[1]
            for i in range(WAY):
                tmp_center = [0.0] * n_features
                for j in range(SHOT):#shot_lists[model_list.index(model_ins)]:#
                    for d in range(n_features):
                        tmp_center[d] += X[i*WAY+j][d]
                for d in range(n_features):
                    tmp_center[d] /= SHOT#len(shot_lists[model_list.index(model_ins)]) #
                center.append(tmp_center)

            avg_var = 0.0
            max_var = -1.0
            min_var = -1.0
            max_idx = -1
            min_idx = -1
            var_list = []
            for i in range(WAY):
                Var = 0.0
                for j in range(SHOT):#shot_lists[model_list.index(model_ins)]:#
                    Var += get_squre(center[i], X[i*WAY + j], dim=n_features) / n_features
                Var /= SHOT#len(shot_lists[model_list.index(model_ins)])#
                var_list.append(Var)
                avg_var += Var
                if max_var == -1.0 or Var > max_var:
                    max_var = Var
                    max_idx = i
                if min_var == -1.0 or Var < min_var:
                    min_var = Var
                    min_idx = i
                #print(f"{dataset_ins} {model_ins[1]} Var({i}): {Var}")
            
            plot_dict[f"{dataset_ins}-{model_ins[1]}-var"] = var_list
            avg_var /= WAY
            
            sorted_var_list = []
            for v in var_list:
                sorted_var_list.append(v)
            sorted_var_list.sort()
            median_var = sorted_var_list[len(sorted_var_list)//2]
            ten_id_list = [-1] * 10
            ten_var_list = [sorted_var_list[i*11] for i in range(10)]
            for i in range(len(var_list)):
                if var_list[i] in ten_var_list:
                    ten_id_list[ten_var_list.index(var_list[i])] = i 
            
            res_dict[f"{dataset_ins}-{model_ins[1]}-avg_var"] = avg_var
            res_dict[f"{dataset_ins}-{model_ins[1]}-min_var"] = min_var
            res_dict[f"{dataset_ins}-{model_ins[1]}-max_var"] = max_var
            res_dict[f"{dataset_ins}-{model_ins[1]}-median_var"] = median_var
            res_dict[f"{dataset_ins}-{model_ins[1]}-min_idx"] = min_idx
            res_dict[f"{dataset_ins}-{model_ins[1]}-max_idx"] = max_idx
            res_dict[f"{dataset_ins}-{model_ins[1]}-ten_id_list"] = ten_id_list
            res_dict[f"{dataset_ins}-{model_ins[1]}-ten_var_list"] = ten_var_list

            dis_list = []
            min_dis = -1.0
            max_dis = -1.0
            min_pair = (-1, -1)
            max_pair = (-1, -1)
            avg_dis = 0.0
            for i in range(len(center)):
                for j in range(i + 1, len(center)):
                    dis = get_distance(center[i], center[j], dim=n_features) / n_features
                    avg_dis += dis
                    dis_list.append(dis)
                    if min_dis == -1.0 or dis < min_dis:
                        min_dis = dis
                        min_pair = (i, j)
                    if max_dis == -1.0 or dis > max_dis:
                        max_dis = dis
                        max_pair = (i, j)
                    #print(f"{dataset_ins} {model_ins[1]} Distance({i},{j}): {dis}")
            plot_dict[f"{dataset_ins}-{model_ins[1]}-dis"] = dis_list
            avg_dis /= len(dis_list)
            
            
            sorted_dis_list = []
            for v in dis_list:
                sorted_dis_list.append(v)
            sorted_dis_list.sort()            
            median_dis = sorted_dis_list[len(sorted_dis_list)//2]

            
            ten_pair_list = [(-1, -1)] * 10
            ten_dis_list = [sorted_dis_list[min(i*550, len(sorted_dis_list)-1)] for i in range(10)]
            cnt = 0
            for i in range(len(center)):
                for j in range(i+1, len(center)):
                    curr = dis_list[cnt]
                    cnt += 1
                    if curr in ten_dis_list:
                        ten_pair_list[ten_dis_list.index(curr)] = (i, j)

            dis_list.sort()
            res_dict[f"{dataset_ins}-{model_ins[1]}-avg_dis"] = avg_dis
            res_dict[f"{dataset_ins}-{model_ins[1]}-min_dis"] = min_dis
            res_dict[f"{dataset_ins}-{model_ins[1]}-max_dis"] = max_dis
            res_dict[f"{dataset_ins}-{model_ins[1]}-median_dis"] = median_dis
            res_dict[f"{dataset_ins}-{model_ins[1]}-min_pair"] = min_pair
            res_dict[f"{dataset_ins}-{model_ins[1]}-max_pair"] = max_pair
            res_dict[f"{dataset_ins}-{model_ins[1]}-ten_pair_list"] = ten_pair_list
            res_dict[f"{dataset_ins}-{model_ins[1]}-ten_dis_list"] = ten_dis_list

    with open('stat.json', 'w') as f:
        json.dump(res_dict, f)
    with open('stat_plot.json', 'w') as f:
        json.dump(plot_dict, f)
exit(1)
fig, ax =  plt.subplots(nrows=len(dataset_list), ncols=2, figsize=(10*2, 10*len(dataset_list)))
for i in range(len(dataset_list)):
    for j in range(len(model_list)):
        var_list = plot_dict[f"{dataset_list[i]}-{model_list[j][1]}-var"]
        print(np.array(var_list).shape)
        ax[i, 0].plot(np.array(range(len(var_list))), np.array(var_list), 'o-', color=f"C{j}", label=f"{model_list[j][1]}")
        ax[i, 0].set_xlabel('class_id')
        ax[i, 0].set_ylabel('variance\n(radius)')
        ax[i, 0].set_title(f'n-dim sphere radius per class ({dataset_list[i]})')
        ax[i, 0].legend()

        dis_list = plot_dict[f"{dataset_list[i]}-{model_list[j][1]}-dis"]
        ax[i, 1].plot(np.array(range(len(dis_list))), np.array(dis_list), 'o-', color=f"C{j}", label=f"{model_list[j][1]}")
        ax[i, 1].set_xlabel('class_pair_id')
        ax[i, 1].set_ylabel('distance')
        ax[i, 1].set_title(f'Center distankce per class pair ({dataset_list[i]})')
        ax[i, 1].legend()
fig.savefig('stat.png')