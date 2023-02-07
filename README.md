# FedNets : Federated Learning on Edge Devices using Ensembles of Pruned Deep Neural Networks

This repository contains the code and experiments for the manuscript:

###### add abstract here

# Installation
- Create a virtual environment with conda/virtualenv.
- Clone the repo.
- Run: `cd <PATH_TO_THE_CLONED_REPO>`.
- Run: `pip3 install -r requirements.txt` to install necessary packages and path links.

# Reproduce Paper Results on Federated CIFAR100 dataset
## Ensemble Generation
- Run: `cd <PATH_TO_THE_CLONED_REPO>/ensyth_pool/pruning/`.
- Run: `python generate_hyperparams.py`; this will generate a CSV file 'constant_spar_hyperparams.csv'.
- Run: `python constant_sparsity.py`; this will generate a pool of pruned models in 'pruned_models' folder. 
## Ensemble Pruning
- Run: `cd <PATH_TO_THE_CLONED_REPO>/ensyth_pool/clustering/`.
- Run: `python generate_cluster.py`; the output of this command is a CSV file 'acc_results.csv'
- Run `python cluster_acc_createFolders.py` to generate a folder that divides the pruned models into different folders (based on the resulted clusters). The number of folders refer to the number of clusters.

