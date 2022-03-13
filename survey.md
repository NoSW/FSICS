# Survey of FSL

## 1. Introduction to Datasets of FSL

`/dataset/README.md`

## 2. Type of FSL approaches

* Data Augmentation Methods
* Metrics Based Methods
* Models Based Methods
* Optimization Based Methods
* Others

### 2.1 Data Augmentation Methods

**Desc:** enhance amount and enrich quality of train datasets

**Well known methods:**

* Image Augmentation Methods
  * geometric transformations
  * color space augmentations
* GANs (Generative Adversarial Networks)

**Cons:**

* skewed data distribution will result in  augmented data distribution to be skewed as well
*  high chance of over-fitting with data augmentation

### 2.2 Metrics Based Methods

**Desc**: calculate similarity between 2 images, e.g. Euclidean distance or cosine similarity

**Well known methods:**

* Siamese Networks(2015, [link](https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf))
* Matching Networks(2017, [link](https://arxiv.org/abs/1606.04080))

### 2.3 Models Based Methods

### 2.4 Optimization Based Methods

### 2.5 Others