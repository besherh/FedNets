import tensorflow as tf
import tensorflow_federated as tff
import numpy as np
import pandas as pd

cifar100_train,cifar100_test = tff.simulation.datasets.cifar100.load_data()
print(cifar100_train.client_ids)
'''
client_local_dataset = cifar100_train.create_tf_dataset_for_client('0')
X = np.array([elem["image"] for elem in client_local_dataset])
y = np.array([elem["label"] for elem in client_local_dataset])
x_train, y_train = X[:70], y[:70]
x_test, y_test  = X[70:], y[70:]

x_train, x_test = x_train.astype('float32') / 255 , x_test.astype('float32') / 255

x_train -= np.mean(x_train, axis=0)
x_test -= np.mean(x_test, axis=0)

y_train, y_test = tf.keras.utils.to_categorical(y_train , 100), tf.keras.utils.to_categorical(y_test , 100)

print(x_train.shape, y_train.shape)
print(x_test[0], y_test[0])
model = tf.keras.models.load_model('./baseline/cifar100_ResNet20v1_model.127.h5')
model.compile("adam", "categorical_crossentropy", metrics=["accuracy"])
history = model.fit(x_train,y_train,verbose=1, epochs = 10, validation_data = (x_test, y_test) )
'''