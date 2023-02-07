# FedNets : Federated Learning on Edge Devices using Ensembles of Pruned Deep Neural Networks

This repository contains the code and experiments for the manuscript.

###### add abstract here(updated abstract goes here)
## Accuracy Results goes here ( add the figures from the paper)

# Installation
- Create a virtual environment with conda/virtualenv.
- Clone the repo.
- Run: `cd <PATH_TO_THE_CLONED_REPO>`.
- Run: `pip3 install -r requirements.txt` to install necessary packages and path links.

# Reproduce Paper Results on Federated CIFAR100 dataset
## Ensemble Generation
- Run: `cd <PATH_TO_THE_CLONED_REPO>/ensyth_pool/pruning/`.
- Create a new folder 'pruned_models'.
- Run: `python generate_hyperparams.py`; this will generate a CSV file 'constant_spar_hyperparams.csv'.
- Run: `python constant_sparsity.py`; this will generate a pool of pruned models in 'pruned_models' folder. 
- Copy the the folder 'pruned_models' to `<PATH_TO_THE_CLONED_REPO>/ensyth_pool/clustering/`.

## Ensemble Pruning
- Run: `cd <PATH_TO_THE_CLONED_REPO>/ensyth_pool/clustering/`.
- Create a new folder 'cluster_folder'.
- Run: `python generate_cluster.py`; the output of this command is a CSV file 'acc_results.csv'
- Run `python cluster_acc_createFolders.py` to generate a folder that divides the pruned models into different folders (based on the resulted clusters). The number of folders refers to the number of clusters.
- Copy the content of the following folder: 'cluster_folder' to `<PATH_TO_THE_CLONED_REPO>/scripts/clients/`.
## FedNets Generation
- Run: `cd <PATH_TO_THE_CLONED_REPO>/scripts/`.
- Run: `python edge_client_api.py`
- Open a new terminal (make sure that the virtualenv is activated and you still in `<PATH_TO_THE_CLONED_REPO>/scripts/`.
- Run `./main_commands.sh`. The output of the scripts is stored in 'python_output.txt'



