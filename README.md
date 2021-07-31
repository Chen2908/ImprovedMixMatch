# ImprovedMixMatch
_______________________________________________________________________________________________________________________________________________________________________

#### Submitted by Chen Galed 311123541 & Hilla Ben-Avi 312545155
_______________________________________________________________________________________________________________________________________________________________________

This is the imlementation of the final project in Machine Learning course, 2021.
This code was writen in python 3.8 using tenforflow 2.2.0.


We chose the algorithm MixMatch presented in the original paper ["MixMatch: A Holistic Approach toSemi-Supervised Learning"](https://arxiv.org/abs/1905.02249).
The original implementation of the algorithm was inspired by the code published in [mixmatch-tensorflow2.0](https://github.com/ntozer/mixmatch-tensorflow2.0).
We suggested an improvement to this algorithm which involves expanding the variety of augmentations applied on the images dataset.
In addition, we implemented a baseline algorithm using transfer learning from ResNet pretrained network.

### Datasets

All three algorithms were tested on 20 images datasets created by splitting the following benchmarks in a stratified manner: 
CIFAR-10 and CIFAR-100 [[1]](#1), SVHN [[2]](#2), and STL-10 [[3]](#3).
An example for a dataset is included in this repository under the title *SVHN_partial_ds_0*.
_______________________________________________________________________________________________________________________________________________________________________

### How to run the code

To run a the main loop simply insert the dataset name you wish to run the code for using the main argument *dataset*. 
***For example: --dataset=SVHN_partial_ds_0***

The run the main.py file from the project directory.
```python3 main.py

_______________________________________________________________________________________________________________________________________________________________________


### References:

<a id="1">[1]</a> 
A. Krizhevsky, "Learning multiple layers of features from tiny images," University of Toronto, Toronto, 2009

<a id="2">[2]</a> 
Y. Netzer, T. Wang, A. Coates, . A. Bissacco and B. W, "Reading digits in natural images with unsupervised feature learning," in NIPS Workshop on Deep Learning and Unsupervised Feature Learning, 2011

<a id="3">[3]</a> 
A. Coates, A. Ng and H. Lee, "An analysis of single-layer networks in unsuper-vised feature learning.," Proceedings of the fourteenth international conference on artificialintelligence and statistics, p. 215–223, 2011
