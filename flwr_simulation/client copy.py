import argparse
import os

import numpy as np
import tensorflow as tf
import tensorflow_federated as tff
import flwr as fl

# Make TensorFlow logs less verbose

#export TF_CPP_MIN_LOG_LEVEL=3

# Define Flower client
class CifarClient(fl.client.NumPyClient):
    def __init__(self, model, x_train, y_train, x_test, y_test, client_id):
        self.id = client_id
        self.model = model
        self.x_train, self.y_train = x_train, y_train
        self.x_test, self.y_test = x_test, y_test

    def get_parameters(self):
        """Get parameters of the local model."""
        raise Exception("Not implemented (server-side parameter initialization)")

    def fit(self, parameters, config):
        """Train parameters on the locally held training set."""

        # Update local model parameters
        self.model.set_weights(parameters)

        # Get hyperparameters for this round
        #batch_size = int(config["batch_size"])
        #Cepochs = int(config["local_epochs"])

        # Train the model using hyperparameters from config
        history = self.model.fit(
            self.x_train,
            self.y_train,
           # batch_size,
            epochs = 10
        )

        # Return updated model parameters and results
        parameters_prime = self.model.get_weights()
        num_examples_train = len(self.x_train)
        results = {
            "loss": history.history["loss"][0],
            "accuracy": history.history["accuracy"][0],
            "val_loss": history.history["val_loss"][0],
            "val_accuracy": history.history["val_accuracy"][0],
        }
        return parameters_prime, num_examples_train, results

    def evaluate(self, parameters, config):
        """Evaluate parameters on the locally held test set."""
        print("------------------  evaluate clients -------------------------------")
        # Update local model with global parameters
        self.model.set_weights(parameters)

        # Get config values
        steps: int = config["val_steps"]

        # Evaluate global model parameters on the local test data and return results
        loss, accuracy = self.model.evaluate(self.x_test, self.y_test, 32, steps=steps, verbose=0)
        num_examples_test = len(self.x_test)
        print("{0}: Testing accuracy is: {1}".format(self.id, accuracy))
        with open('./fd.csv','a') as file:
            file.writelines("{0}: Testing accuracy is: {1}\n".format(self.id, accuracy))
        return loss, num_examples_test, {"accuracy": accuracy}


def main() -> None:
    # Parse command line argument `partition`
    parser = argparse.ArgumentParser(description="Flower")
    #parser.add_argument("--partition", type=int, choices=range(0, 10), required=True)
    parser.add_argument("--partition", type=int,  required=True)
    args = parser.parse_args()

    # Load and compile Keras model
    model = tf.keras.models.load_model('./baseline/cifar100_ResNet20v1_model.127.h5')
    model.compile("adam", "categorical_crossentropy", metrics=["accuracy"])

    # Load a subset of CIFAR-100 to simulate the local data partition
    (x_train, y_train), (x_test, y_test) = load_partition(args.partition)

    # Start Flower client
    client_id = str(args.partition)
    client = CifarClient(model, x_train, y_train, x_test, y_test, client_id)
    fl.client.start_numpy_client("localhost:8080", client=client)


def load_partition_old(idx: int):

    cifar100_train,cifar100_test = tff.simulation.datasets.cifar100.load_data()
    test_local_dataset = cifar100_test.create_tf_dataset_for_client(str(idx))
    train_local_dataset = cifar100_train.create_tf_dataset_for_client(str(idx))

    x_test = np.array([elem["image"] for elem in test_local_dataset])
    y_test = np.array([elem["label"] for elem in test_local_dataset])
    x_train = np.array([elem["image"] for elem in train_local_dataset])
    y_train = np.array([elem["label"] for elem in train_local_dataset])

    # it's always better to normalize
    x_test = x_test.astype('float32') / 255
    x_train = x_train.astype('float32') / 255
    x_test_mean = np.mean(x_test, axis=0)
    x_train_mean = np.mean(x_train, axis=0)

    x_test -= x_test_mean
    x_train -= x_train_mean
    # one hot 
    y_test = tf.keras.utils.to_categorical(y_test , 100)
    y_train = tf.keras.utils.to_categorical(y_train , 100)
    
    return (x_train,y_train),(x_test,y_test)

def load_partition(idx: int):
    cifar100_train,cifar100_test = tff.simulation.datasets.cifar100.load_data()
    #print(cifar100_train.client_ids)
    client_local_dataset = cifar100_train.create_tf_dataset_for_client(str(idx))
    X = np.array([elem["image"] for elem in client_local_dataset])
    y = np.array([elem["label"] for elem in client_local_dataset])
    x_train, y_train = X[:70], y[:70]
    x_test, y_test  = X[70:], y[70:]

    x_train, x_test = x_train.astype('float32') / 255 , x_test.astype('float32') / 255
    x_train -= np.mean(x_train, axis=0)
    x_test -= np.mean(x_test, axis=0)

    y_train, y_test = tf.keras.utils.to_categorical(y_train , 100), tf.keras.utils.to_categorical(y_test , 100)        
    return (x_train,y_train),(x_test,y_test)


if __name__ == "__main__":
    main()