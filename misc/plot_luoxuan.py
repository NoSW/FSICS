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
def plot_axes(four, dataset, ax):
    new = []
    for v in four['plot_acc_new']:
        new.append(v if v > 0.5 else new[-1])
    old = []
    for v in four['plot_acc_old']:
        old.append(v if v > 0.5 else old[-1])
 
    ax.plot(np.array(four['plot_x']), np.array(four['plot_acc_blank']), 'o-', color=f"C0", label=f"control classifier")
    ax.plot(np.array(four['plot_x']), np.array(new), 'o-', color=f"C1", label=f"classifier A")
    ax.plot(np.array(four['plot_x']), np.array(old), 'o-', color=f"C2", label=f"classifier B")
    ax.set_title(f'accuracy on query set ({dataset}, 5way-5shot, ViT-Small)')
    ax.set_xlabel('epoch')
    ax.set_ylabel('acc')
    if dataset == 'FC100':
        ax.set_ylim(bottom=0.6)
    else:
        ax.set_ylim(bottom=0.82)
    ax.legend()


fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10*2, 10))
for dataset in dataset_list:
    _d = read_json(f'gan\\{dataset}-luoxaun-2.json')
    plot_axes(_d, dataset, axs[dataset_list.index(dataset)])
fig.savefig('zluoxaun-5way-5shot.png')
