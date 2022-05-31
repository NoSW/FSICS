# support init or not
from traceback import print_tb
import numpy as np
import matplotlib.pyplot as plt
import json
from os.path import join

plt.style.use([ 'grid','notebook',])
plt.rc('text', usetex=True)

def read_json(fn):
    with open(fn, 'r') as f:
        return json.loads(f.read())


dataset_list = [
        "mini-imagenet",
        "FC100",
    ]
ARCH =  "mocov3-vit_small"
way_shot_dict ={
        # way [shot0, shot1, ...]
        2: [1, 5, 10, 20],
        5: [1, 5, 10, 20],
        10: [1, 5, 10, 20],
        50: [1, 5, 10, 20],
    }
def plot_axes(backbone, four, tran, dataset, ax, way, shot):
    #ax.plot(np.array(backbone['plot_x']), np.array(backbone['plot_loss_y']), 'o-', color=f"C0", label=f"random ")
    ax.plot(np.array(four['x']), np.array(four['loss_blank']), 'o-', color=f"C0", label=f"without entropy regularization")
    ax.plot(np.array(four['x']), np.array(four['loss_zero']), 'o-', color=f"C1", label=f"with entropy regularization")
    #ax.plot(np.array(tran['plot_x']), np.array(tran['new_loss_y']), 'o-', color=f"C2", label=f"inherited")
    ax.set_title(f'{dataset} ({way}way-{shot}shot, ViT-Small)')
    ax.set_xlabel('epoch')
    ax.set_ylabel('loss')
   # ax.set_ylim(0.0, 1.0)
    ax.legend()

TRANS_FLODER = 'result\TRANS_5'
BACKBONE_FLODER = 'result\BACKBONE_SMALL_5'
FOUR_FLODER = 'result\FOUR5'
COMP_FLODER = 'result\BACKBONE_TTTTTTTTTT'

def avg_dict(fn_prefix, cnt):
    res_dict = {}
    for i in range(cnt):
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


fig, axs = plt.subplots(nrows=1, ncols=4, figsize=(10*4, 10))
#for dataset in dataset_list:
    #backbone_dict = avg_dict(join(BACKBONE_FLODER, f'{dataset}-5way-5shot'))
    #tran_dict = avg_dict(join(TRANS_FLODER, f'{dataset}-5way-1shot'))
way_list = [2,5,10,50]
for i in range(len(way_list)):
    way = way_list[i]
    four_dict = avg_dict(join(f'result\FOUR{way}', f'FC100-vit_small-{way}way-5shot'), cnt= (1 if way==50 else 10))
    plot_axes(None, four_dict, None, 'FC100', axs[i], way, 5)
fig.savefig('entropy-mway-5shot.png')
