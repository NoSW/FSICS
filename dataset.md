# Data-Sets for Few-Shot Learning

## mini-ImageNet

The mini-ImageNet dataset was proposed by Vinyals et al. for few-shot learning evaluation. Its complexity is high due to the use of ImageNet images but requires fewer resources and infrastructure than running on the full ImageNet dataset. In total, **there are 100 classes with 600 samples of 84×84 color images per class.** These 100 classes are divided into 64, 16, and 20 classes respectively for sampling tasks for meta-training, meta-validation, and meta-test. 

Introduced by Vinyals et al. in [Matching Networks for One Shot Learning](https://arxiv.org/abs/1606.04080) (2017)

## tiered-ImageNet

 The **tiered-ImageNet** dataset is a larger subset of ILSVRC-12 with **608 classes (779,165 images) grouped into 34 higher-level nodes** in the ImageNet human-curated hierarchy. This set of nodes is partitioned into 20, 6, and 8 disjoint sets of training, validation, and testing nodes, and the corresponding classes form the respective meta-sets. As argued in Ren et al. (2018), this split near the root of the ImageNet hierarchy results in a more challenging, yet realistic regime with test classes that are less similar to training classes. 

Introduced by Ren et al. in [Meta-Learning for Semi-Supervised Few-Shot Classification](https://paperswithcode.com/paper/meta-learning-for-semi-supervised-few-shot) (2018)

## Fewshot-CIFAR100

 The **FC100** dataset (**Fewshot-CIFAR100**) is a newly split dataset based on CIFAR-100 for few-shot learning. The splits were proposed by TADAM. It contains **20 high-level categories** which are divided into 12, 4, 4 categories for training, validation and test. There are **60, 20, 20 low-level classes** in the corresponding split containing **600 images of size 32 × 32 per class**. Smaller image size makes it more challenging for few-shot learning. 

Introduced by Oreshkin et al. in [TADAM: Task dependent adaptive metric for improved few-shot learning](https://paperswithcode.com/paper/tadam-task-dependent-adaptive-metric-for) (2018)

## CIFAR-FS

 **CIFAR100 few-shots** (**CIFAR-FS**) is randomly sampled from CIFAR-100 (Krizhevsky & Hinton, 2009) by using the same criteria with which mini-ImageNet has been generated. The average inter-class similarity is sufficiently high to represent a challenge for the current state of the art. Moreover, the limited original resolution of 32×32 makes the task harder and at the same time allows fast prototyping. 

Introduced by Bertinetto et al. in [Meta-learning with differentiable closed-form solvers](https://paperswithcode.com/paper/meta-learning-with-differentiable-closed-form) (2019)



## Omniglot

The Omniglot dataset is designed for developing more human-like learning algorithms. It contains 1623 different handwritten characters from 50 different alphabets. The data-set is divided into a background set and an evaluation set. Background set contains 30 alphabets (964 characters) and only this set should be used to perform all learning (e.g. hyper-parameter inference or feature learning). The remaining 20 alphabets are for pure evaluation purposes only. Each character is a 105 x 105 greyscale image. There are only 20 samples for each character, each drawn by a distinct individual

 Each of the 1623 characters was drawn online via Amazon's Mechanical Turk by 20 different people. Each image is paired with stroke data, a sequences of [x,y,t] coordinates with time (t) in milliseconds. 

Introduced by Brenden et al. in [Human-level concept learning through probabilistic program induction](https://www.science.org/doi/abs/10.1126/science.aab3050) (2015) [github](https://github.com/brendenlake/omniglot)

## Reference

[Few-Shot Classification Leaderboard](https://fewshot.org/miniimagenet.html), a project keeping on track with the state-of-the-art for the few-shot classification.

[papers with code (datasets)](https://paperswithcode.com/datasets), a website collecting papers