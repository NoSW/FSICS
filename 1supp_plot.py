# support init or not
import numpy as np
import matplotlib.pyplot as plt

plt.style.use([ 'grid','notebook',])
plt.rc('text', usetex=True)


def read_txt(fn):
    res = {}
    with open(fn, 'r') as f:
        for line in f.readlines(): 
            tokens = line.split(' ')
            acc = float(tokens[1][:8])
            res[tokens[0]] = acc
    return res

dataset_list = [
        "mini-imagenet",
        "FC100",
        # "tiered_imagenet",
    ]
ARCH = "mocov3-vit_small"
way_shot_dict ={
        # way [shot0, shot1, ...]
        2: [1, 5, 10, 20],
        5: [1, 5, 10, 20],
        10: [1, 5, 10, 20],
        50: [1, 5, 10, 20, 50],
        100: [1, 5, 10, 20, 50],
    }

def plot_axes(yes_res, no_res, dataset_ins, ax, way):
    y_no = []
    y_yes = []
    y = []
    for shot_ins in way_shot_dict[way]:
        k = f"{dataset_ins}-{ARCH}-{way}way-{shot_ins}shot"
        y_no.append(no_res[k])
        y_yes.append(yes_res[k])
    ax.plot(np.array(way_shot_dict[way]), np.array(y_no), 'o-', color=f"C0", label=f"random {way}way random")
    ax.plot(np.array(way_shot_dict[way]), np.array(y_yes), 'o-', color=f"C1", label=f"support-based {way}way random")
    ax.set_title(f'weight initialization ({dataset_ins})')
    ax.set_xlabel('shot')
    ax.set_ylabel('accuracy')
   # ax.set_ylim(0.0, 1.0)
    ax.legend()


fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(10, 14))
yes_res = read_txt("result\\SUPP002_2022-04-27-05-09-24-4590\\a-result-2022-04-27-05-09-24-4590.txt")
no_res = read_txt("result\\RAND002_2022-04-27-13-30-00-956\\a-result-2022-04-27-13-30-00-956.txt")
plot_axes(yes_res, no_res, "mini-imagenet", axs[0], 2)
plot_axes(yes_res, no_res, "FC100", axs[1], 2)
fig.savefig('support.png')
