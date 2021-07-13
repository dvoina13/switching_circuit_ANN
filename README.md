## A biologically inspired architecture with switching units is capable of context-dependent continual learning
switching networks for context-dependent continual learning

This repository is the official implementation of "A biologically inspired architecture with switching units is capable of context-dependent continual learning".



# Requirements:

Two anaconda environments are provided:
environment1.yml and environment2.yml. The first environment was usedfasfasfasfasfasfsafaf to train and evaluate the basic network and switching network with 2, 3, and 4 hidden layers, while the second environment was used to train and evaluate VGG-16.

`conda env create -f environment.yml`

# Training:
    
Training and evaluation are provided in the same scripts: train_fig2.py, train_fig3.py, train_fig5.py, Suppl_NN_wSwitchDetector.py, SupplFig_otherSimple_nets.py, MNIST+c_on_VGG.py, MNIST+c+switch_on_VGG.py

To run training and evaluation run the following .sh files:

run_experiment.sh - for training and evaluation to generate Figures 2,3,5 (and for some Supplementary Figures)

run_suppl.sh - for training and evaluation to generate Suppl. Figs. 3 and 10\

run_vgg16.sh - for training and evaluation of the VGG-16 networks (with or without switching units) - Figures 2,3, Suppl. Figs. 1, 11


`./run_experiment.sh`

# Results:
    
see Figures 2,3,4,5 of the paper

# References:
"A biologically inspired architecture with switching units is capable of context-dependent continual learning", 2021

Our code is under the MIT license (see LICENSE)
