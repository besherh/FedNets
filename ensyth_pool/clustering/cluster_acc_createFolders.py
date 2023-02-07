import os
from tokenize import Double 
import pandas as pd
import numpy as np
import glob
from shutil import copyfile
import shutil
from typing import KeysView

parent_dir = "./cluster_folder/"
pruning_folder = "./pruned_models/"

def file_browser(directory_name):
  return [model for model in glob.glob(directory_name + "*.h5")]

def get_file_name(file_path):
    my_path = str(file_path)
    return my_path.split(os.sep)[-1]


def acc_first(input_ds):
    acc_dic = {}
    cluster_column = input_ds["cluster"]
    k = cluster_column.max()
    for i in range(0,k):
        temp_ds = input_ds[input_ds["cluster"]== i]
        acc_dic[i] = temp_ds["acc"].mean()
    print(acc_dic)
    print("the shape after merging is:{0}".format(input_ds.shape))
    series = pd.Series(acc_dic)
    best_cluster_idx = series.idxmax()

    print(best_cluster_idx)


    return input_ds[input_ds["cluster"] == best_cluster_idx]


def main():
    cluster_ds = pd.read_csv("clustering_results.csv",names=['model', 'cluster'])
    acc_ds = pd.read_csv("acc_results.csv",names = ['model','acc'])

    merged_ds = pd.merge(cluster_ds,acc_ds, on='model')

    filtred_ds = merged_ds[merged_ds["acc"]>=0.30]
    print("number of models after filtering on acc:", filtred_ds.shape)
    df_shuffled = filtred_ds.sample(frac=1)
    clients = np.array_split(df_shuffled,10)
    if os.path.exists(parent_dir):
        shutil.rmtree(parent_dir)
    else:
        os.mkdir(parent_dir) 
    i =1
    for client in clients:        
        os.mkdir(os.path.join(parent_dir,str(i)))
        for model in client["model"]:
            source = str(model)
            destination = os.path.join(parent_dir,str(i),get_file_name(model))
            copyfile(source, destination)        
        i = i+1

      

if __name__ =="__main__":
    main()
