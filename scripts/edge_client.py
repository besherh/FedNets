
import numpy as np
import tensorflow as tf
import tensorflow_federated as tff
import tensorflow_datasets  as tfds
import flask
from tensorflow.keras.models import load_model
import json
import glob
import pandas as pd
from stellargraph import StellarGraph
from threading import Thread
import helpers
import os
os. environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class Edge_Client:
    def __init__(self,client_id, files_dir, data_set):
        self.id = client_id
        self.data_set = data_set
        self.models_dir = files_dir
        self.models_list = [model for model in glob.glob(self.models_dir + "*.h5")]
        self.subtract_pixel_mean = True
        self.number_of_classes = 0
        self.batch_size = 32
        self.epochs = 10
        if data_set == 'cifar100':
                cifar100_train,cifar100_test = tff.simulation.datasets.cifar100.load_data()
                #print(cifar100_train.client_ids)
                client_local_dataset = cifar100_train.create_tf_dataset_for_client(client_id)
                X = np.array([elem["image"] for elem in client_local_dataset])
                y = np.array([elem["label"] for elem in client_local_dataset])
                x_train, y_train = X[:70], y[:70]
                x_test, y_test  = X[70:], y[70:]
                x_train, x_test = x_train.astype('float32') / 255 , x_test.astype('float32') / 255
                x_train -= np.mean(x_train, axis=0)
                x_test -= np.mean(x_test, axis=0)
                y_train, y_test = tf.keras.utils.to_categorical(y_train , 100), tf.keras.utils.to_categorical(y_test , 100)        
                self.x_train = x_train
                self.y_train = y_train
                self.x_test = x_test
                self.y_test = y_test
        else:
            raise Exception("Not implemented yet")

    def generate_validation_acc(self,validation_ratio):            
            if self.data_set == 'cifar100':
                print('Intialising training cifar100 federated dataset, my client id is :{0}'.format(self.id))
                self.number_of_classes = 100
                number_of_samples = int(float(validation_ratio) * self.x_train.shape[0])
                print("writing the accuracy!")
                file_name = "{0}/validation_acc.csv".format(self.models_dir)
                with open(file_name, 'w') as txt_file :
                    for model in self.models_list:
                        print(model)
                        loaded_model = load_model(model)
                        opt, loss = helpers.get_model_metdata(model)
                        loaded_model.compile(optimizer=opt,loss = loss, metrics=['accuracy'])
                        scores = loaded_model.evaluate(self.x_train[:number_of_samples], self.y_train[:number_of_samples], verbose=1)
                        acc = scores[1]
                        txt_file.write("{0},{1:.2f}\n".format(self.models_dir+helpers.path_leaf(model),acc))    
                          
            else:
                raise Exception("Not implemented yet")

    
    def get_model_yhats(self,model_url, x_test):
        loaded_model = load_model(model_url)
        y_pred = loaded_model.predict(x_test)
        return y_pred
    
    def train_local_models(self,x_train,y_train):        
        for model in self.models_list:
            print('Training started in client: {0}, model {1}'.format(self.id,model))
            loaded_model = load_model(model)            
            opt, loss = helpers.get_model_metdata(model)
            print('Optimiser: {0}, Loss: {1}'.format(opt,loss))
            loaded_model.compile(optimizer=opt,loss = loss, metrics=['accuracy'])
            loaded_model.fit(x_train,y_train,batch_size=self.batch_size,
            epochs=self.epochs)


    def get_ensemble_predictions(self, x_test):
        print('generating ensemble predictions')
        model_pool_prediction = {}
        print('Total models in the folder: {0}'.format(len(self.models_list)))
        for model_name in self.models_list:
                model_pool_prediction[model_name] = self.get_model_yhats(model_name,x_test)
        yhats = self.combine_predictions(len(x_test), model_pool_prediction)
        return yhats    

    def combine_predictions(self,test_samples, model_pool_dic):
        i = 0
        yhats = []
        while i <= test_samples-1:
            one_instance_prediction = []
            for key in model_pool_dic:
                one_instance_prediction.append(model_pool_dic[key][i])
            temp = np.array(one_instance_prediction)
            yhat = np.sum(temp, axis=0)
            yhats.append(yhat)  
            i += 1
        return np.array(yhats)

    def calculate_ensemble_acc(self,y_pred,x_test,y_test):
        total_seen = 0
        total_correct = 0
        for i in range(0,x_test.shape[0]):
            total_seen += 1
            if(np.argmax(y_test[i]) == np.argmax(y_pred[i])):
                total_correct +=1
        print("My id is:", self.id)
        print("total correct", total_correct)
        print("total seen", total_seen)
        print("ensemble acc :",float(total_correct) / float(total_seen) )
        return float(total_correct) / float(total_seen)
 


