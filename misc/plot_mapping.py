from traceback import print_tb
import numpy as np
import matplotlib.pyplot as plt
import json
from os.path import join

from pandas import array

plt.style.use([ 'grid','notebook',])
plt.rc('text', usetex=True)

def read_json(fn):
    with open(fn, 'r') as f:
        return json.loads(f.read())


x = np.array([2, 5, 10, 20])

fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(10*3, 10))


non = np.array([0.817417,0.905649,0.957559,0.975474 ])
best =np.array( [0.80356,0.939798,0.96932,0.975332]) #(acc=0.9789)
ax[0].plot(x, non, 'o-', color="C0", label="without mapping")
ax[0].plot(x, best, 'o-', color="C1", label="with mapping")
ax[0].set_title("2way, k-shot")
ax[0].set_ylabel('acc')
ax[0].set_xlabel('shot')
ax[0].legend()


non = np.array([0.649643,0.838004,0.877970,0.955200 ])
best =np.array( [0.408734,0.829798,0.919932,0.969932]) #(acc=0.9789)
ax[1].plot(x, non, 'o-', color="C0", label="without mapping")
ax[1].plot(x, best, 'o-', color="C1", label="with mapping")
ax[1].set_title("5way, k-shot")
ax[1].set_ylabel('acc')
ax[1].set_xlabel('shot')
ax[1].legend()


non = np.array([0.574799,0.642664,0.745156,0.832068 ])
best =np.array( [0.535642,0.662664,0.749864,0.869932]) #(acc=0.9789)
ax[2].plot(x, non, 'o-', color="C0", label="without mapping")
ax[2].plot(x, best, 'o-', color="C1", label="with mapping")
ax[2].set_title("10way, k-shot")
ax[2].set_ylabel('acc')
ax[2].set_xlabel('shot')
ax[2].legend()

fig.savefig('mapping-nshot.jpg')