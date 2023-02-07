from ast import Return
import collections

import numpy as np
import tensorflow as tf
import tensorflow_federated as tff
import tensorflow_datasets  as tfds
from flask import Flask  
import helpers
import os
os. environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 21:59:51 2019

@author: Besher
"""
# importing the requests library 
import requests 
import json
import time
from threading import Thread
import logging
from graph_embedder import Graph_Embedder
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

# Global settigns
predict_end_points = ["http://localhost:12345/predict"]
train_end_points = ["http://localhost:12345/train"]
val_end_points = ["http://localhost:12345/validate"]
#ids_list=['0','1','10','11','12','13','14','15','16']
num_of_clients = 7
ids_list = ['0','1']
dir_list = ['./clients/1/','./clients/2/']
dataset = 'cifar100'
validation_ratio = 0.2
acc_threshold = 0.4
cluster_length = 5 
sample_size = 10
threads = []
predictions = []
parent_clients_dir = './clients/'



class Edge_Server(Graph_Embedder):
    def __init__(self, clients_folders,clients_no) -> None:
        super().__init__(clients_folders)
        self.clients_no = clients_no
        self.data_set = dataset
        if self.data_set =='cifar100':
            self.training_set, self.testing_set = tff.simulation.datasets.cifar100.load_data()
            self.client_ids = self.training_set.client_ids[0:num_of_clients]
        elif self.data_set =='emnist':
            self.training_set, self.testing_set = tff.simulation.datasets.emnist.load_data()
            self.client_ids = self.training_set.client_ids[0:num_of_clients]    
    
    def get_prediction_from_edge(self, url, client_id,client_dir, dataset):
        edge_point_params = {'client_id': client_id, 'client_dir':client_dir, 'data_set':dataset }
        response = requests.get(url, edge_point_params)
        if response.status_code == 200:
            logging.info("response true")
            data = response.json()
            data = json.loads(data)        
            predictions.append("client {0}:{1}".format(data['MyId'],data['Accuracy']))

    def perform_local_training_client(self, url, client_id,client_dir, dataset):
        edge_point_params = {'client_id': client_id, 'client_dir':client_dir, 'data_set':dataset }
        response = requests.get(url, edge_point_params)
        if response.status_code == 200:
            logging.info("response true")
            data = response.json()
            data = json.loads(data)     

    def generate_validation_acc(self, url, client_id,client_dir, dataset, validation_ratio):
        edge_point_params = {'client_id': client_id, 'client_dir':client_dir, 'data_set':dataset , 'validation_ratio':validation_ratio }
        response = requests.get(url, edge_point_params)
        if response.status_code == 200:
            logging.info("response true")
            data = response.json()
            data = json.loads(data)     

    def deploy_representative(self,acc_threshold, cluster_length, sample_size):
        df = helpers.acc_csv_to_df(parent_clients_dir)
        df2 = helpers.clustering_csv_to_df("./")
        df3 = helpers.merge_data_frames(df,df2,'model')
        df3_filtered = helpers.df_representative_selection(df3,acc_threshold,cluster_length)
        helpers.generate_new_pool(df3_filtered)
        print("iterating through the clients to deploy new models")
        for client_model in self.clients_folders:
            print(client_model)
            helpers.remove_files_in_dir(client_model)
            helpers.copy_all_file_from_dir('./pool/',client_model,sample_size)
        print("Done - Deploy representatives ")



print("main function")
start_main = time.time()
edge_server = Edge_Server(dir_list, len(ids_list))
#ids_list = edge_server.client_ids


'''
graphs_embeddings = edge_server.generate_embeddings(edge_server.graphs)
cluster_labels = edge_server.cluster_embeddings(graphs_embeddings)
edge_server.deploy_representative(acc_threshold,cluster_length,sample_size)
'''
for i in range(0,len(ids_list)):
        #process = Thread(target=edge_server.perform_local_training_client, args=[train_end_points[0], ids_list[i],dir_list[i], dataset])
        process = Thread(target=edge_server.get_prediction_from_edge, args=[predict_end_points[0], ids_list[i],dir_list[i], dataset])
        #process = Thread(target=edge_server.generate_validation_acc, args=[val_end_points[0], ids_list[i],dir_list[i], dataset, validation_ratio])
        process.start()        
        threads.append(process)
for index, thread in enumerate(threads):
        logging.info("Main    : before joining thread %d.", index)
        thread.join()
        logging.info("Main    : thread %d done", index)

end_main = time.time()
print(predictions)
print("total time:{0}".format(end_main-start_main))

