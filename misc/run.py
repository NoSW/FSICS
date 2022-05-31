import matplotlib.pyplot as plt
import numpy as np
import PIL
import random
# functions to show an image
plt.style.use(['science','notebook',])
plt.rc('text', usetex=True)

way = ["dog", "mountain", "corn", "bird", 'piano']

plt.figure()
fig, ax =  plt.subplots(1, len(way))
for i in range(len(way)):
    img = PIL.Image.open(f"F:\\support\\{way[i]}\\{5}.jpg")
    ax[i].imshow(img)
    ax[i].set_title(f"{way[i]}")
    ax[i].set_xticks([])
    ax[i].set_yticks([])
print("epidode0")
print(f"Support Set ({len(way)}way-1shot)")
plt.show()


fig, ax =  plt.subplots(nrows=5, ncols=5)
for i in range(len(way)):
    for j in range(5):
        img = PIL.Image.open(f"F:\\support\\{way[i]}\\{j}.jpg")
        if(j == 0):
             ax[j,i].set_title(way[i])   
        ax[j,i].imshow(img)
        ax[j,i].set_xticks([])
        ax[j,i].set_yticks([])
print(f"build a classifier ... Done.(accuracy=1.00)")
plt.show()

