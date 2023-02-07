import tensorflow as tf
import pandas as pd
import numpy as np
import glob
import ntpath
import os
import shutil
import random


def find_nth(string, substring, n):
   if (n == 1):
       return string.find(substring)
   else:
       return string.find(substring, find_nth(string, substring, n - 1) + 1)

def file_browser(dir_name):
  return [model for model in glob.glob(dir_name + "*.h5")]

def csv_file_browser(dir_name):
      files= [csv_file for csv_file in glob.glob(dir_name + "*.csv")]
      return files[0]


def get_model_metdata(model_name):
    t = find_nth(model_name,"-",1)
    t2 = find_nth(model_name,"-",2)
    optimizer = model_name[t+1:t2]
    loss = model_name[t2+1:len(model_name)-3]
    return optimizer, loss

def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)

def __get_csvs_from_dir(parent_dir):
    csv_files = []
    for dirpath, dirnames, filenames in os.walk(parent_dir):
        for filename in [f for f in filenames if f.endswith(".csv")]:
            csv_files.append(os.path.join(dirpath, filename))
    return csv_files



def acc_csv_to_df(parent_dir):
    li = []
    all_files = __get_csvs_from_dir(parent_dir)
    for filename in all_files:
        df = pd.read_csv(filename, index_col=None, names=["model","acc"])
        li.append(df)

    frame = pd.concat(li, axis=0, ignore_index=True)
    return frame

def clustering_csv_to_df(parent_dir):
    file_name = csv_file_browser(parent_dir)
    df = pd.read_csv(file_name,index_col=None, names=["model","cluster"])
    return df

def merge_data_frames(df1,df2,key_column):
    return pd.merge(df1, df2, on=key_column, how='inner')

def df_representative_selection(df,acc_trhreshold,cluster_len):
    print("df_representative_selection")
    print(df)
    print("acc:{0},cluster{1}".format(acc_trhreshold,cluster_len))
    df = df.loc[df['acc']>= float(acc_trhreshold)]
    print("df_after acc filtering")
    print(df)
    if df.shape[0] > 15 :
        df = df.groupby("cluster").filter(lambda x: len(x) >= cluster_len)

    print("df_after cluster filtering")
    print(df)
    return df
    #return df.sample(n=sample_size, replace=True)

def generate_new_pool(df):
    print("generating pool...., below are the members")
    print("removing all previous files....")    
    remove_files_in_dir('./pool/')
    for model in df['model']:
        print(model)
        copy_file(model,'./pool/')


def copy_file(src,dest):
    shutil.copy(src, dest)

def copy_all_file_from_dir(src,dest,sample_size):
    files=file_browser(src)
    for i in range(0,sample_size):
        random_file = random.choice(files)
        copy_file(random_file,dest)   

    

def remove_files_in_dir(base_dir):
    files = glob.glob(base_dir+'*')
    for f in files:
        os.remove(f)

  
'''
df = acc_csv_to_df('./clients/')
#print(df)
df2 = clustering_csv_to_df("./")
#print(df2)

df3 = merge_data_frames(df,df2,'model')
print(df3)

df3_filtered = df_representative_selection(df3,0.6,2,3)
print(df3_filtered)

generate_new_pool(df3_filtered)
'''