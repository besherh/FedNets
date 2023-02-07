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
num_of_clients = 10
ids_list = ['0','1','10','11','12','13','14','15','16','17']
dir_list = ['./clients/1/','./clients/2/','./clients/3/','./clients/4/','./clients/5/',
'./clients/6/','./clients/7/','./clients/8/','./clients/9/','./clients/10/']
dataset = 'cifar100'
validation_ratio = 0.2
acc_threshold = 0.4
cluster_length = 5 
sample_size = 10
threads = []
predictions = []
parent_clients_dir = './clients/'
