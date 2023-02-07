import numpy as np
import tensorflow as tf
import tensorflow_federated as tff
import tensorflow_datasets  as tfds
import flask
from tensorflow.keras.models import load_model
import json
import pandas as pd
import helpers
from edge_client import Edge_Client
import os

app = flask.Flask(__name__)
def default(obj):
    if type(obj).__module__ == np.__name__:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj.item()
    raise TypeError('Unknown type:', type(obj))


@app.route("/train", methods=["GET"])
def train_local_models():
    data = {"success": False}
    params = flask.request.json
    if params == None:
        params = flask.request.args
    # if parameters are found, return a prediction
    if (params != None):
        my_id = params['client_id']
        my_dir = params['client_dir']
        my_dataset = params['data_set']
        print('creating edge_client: client details: id:{0}, dir:{1}, dataset:{2}'.format(my_id,my_dir,my_dataset))
        edge_client = Edge_Client(my_id, my_dir, my_dataset)
        #edge_client.intialise_training_non_iid()
        edge_client.train_local_models(edge_client.x_train, edge_client.y_train)
        data = {"success": True}
        dumped = json.dumps(data, default=default)

    # return a response in json format
    return flask.jsonify(dumped)




@app.route("/predict", methods=["GET","POST"])
def predict():
    data = {"success": False}
    params = flask.request.json
    if params == None:
        params = flask.request.args
    # if parameters are found, return a prediction
    if (params != None):
        my_id = params['client_id']
        my_dir = params['client_dir']
        my_dataset = params['data_set']
        print('creating edge_client: client details: id:{0}, dir:{1}, dataset:{2}'.format(my_id,my_dir,my_dataset))
        edge_client = Edge_Client(my_id, my_dir, my_dataset)
        #edge_client.intialise_testing_non_iid()
        print('generating ensemble predictions')
        ensemble_yhats = edge_client.get_ensemble_predictions(edge_client.x_test)
        acc = edge_client.calculate_ensemble_acc(ensemble_yhats,edge_client.x_test,edge_client.y_test)
        data = {"success": True, "Accuracy" : acc, "MyId":my_id}
        dumped = json.dumps(data, default=default)

    # return a response in json format
    return flask.jsonify(dumped)


@app.route("/validate", methods=["GET","POST"])
def generate_val_acc():
    data = {"success": False}
    params = flask.request.json
    if params == None:
        params = flask.request.args
    # if parameters are found, return a prediction
    if (params != None):
        my_id = params['client_id']
        my_dir = params['client_dir']
        my_dataset = params['data_set']
        validation_ratio = params['validation_ratio']
        print('generating edge_client: client details: id:{0}, dir:{1}, dataset:{2}'.format(my_id,my_dir,my_dataset))
        edge_client = Edge_Client(my_id, my_dir, my_dataset)
        edge_client.generate_validation_acc(validation_ratio)
        data = {"success": True}
        dumped = json.dumps(data, default=default)
    # return a response in json format
    return flask.jsonify(dumped)


#app.run(host=getNetworkIp())
app.run('127.0.0.1', port=12345)