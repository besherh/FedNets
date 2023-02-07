from typing import Any, Callable, Dict, List, Optional, Tuple

import flwr as fl
import tensorflow as tf
import tensorflow_federated as tff
import numpy as np

def main() -> None:
    # Load and compile model for
    # 1. server-side parameter initialization
    # 2. server-side parameter evaluation
    model = tf.keras.models.load_model('./baseline/cifar100_ResNet20v1_model.127.h5')
    model.compile("adam", "categorical_crossentropy", metrics=["accuracy"])

    # Create strategy
    strategy = fl.server.strategy.FedAdagrad(
        fraction_fit=1,
        fraction_eval=1,
        min_fit_clients=2,
        min_eval_clients=2,
        min_available_clients=2,
        #eval_fn=get_eval_fn(model),
        #on_fit_config_fn=fit_config,
        on_evaluate_config_fn=evaluate_config,
        initial_parameters=fl.common.weights_to_parameters(model.get_weights()),
    )

    # Start Flower server for four rounds of federated learning
    fl.server.start_server("localhost:8080", config={"num_rounds": 4}, strategy=strategy)


def get_eval_fn(model):
    """Return an evaluation function for server-side evaluation."""

    # Load data and model here to avoid the overhead of doing it in `evaluate` itself
    cifar100_train,cifar100_test = tff.simulation.datasets.cifar100.load_data()
    train_local_dataset = cifar100_train.create_tf_dataset_for_client('102')

    x_val = np.array([elem["image"] for elem in train_local_dataset])
    y_val = np.array([elem["label"] for elem in train_local_dataset])

        # it's always better to normalize
    x_val = x_val.astype('float32') / 255
    x_val_mean = np.mean(x_val, axis=0)

    x_val -= x_val_mean
        # one hot 
    y_val = tf.keras.utils.to_categorical(y_val , 100)
    '''
    # The `evaluate` function will be called after every round
    def evaluate(
        weights: fl.common.Weights,
    ) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
        model.set_weights(weights)  # Update model with the latest parameters
        loss, accuracy = model.evaluate(x_val, y_val)
        return loss, {"accuracy": accuracy}

    return evaluate
    '''

def fit_config(rnd: int):
    """Return training configuration dict for each round.
    Keep batch size fixed at 32, perform two rounds of training with one
    local epoch, increase to two local epochs afterwards.
    """
    config = {
        "batch_size": 32,
        "local_epochs": 1 if rnd < 2 else 2,
    }
    return config


def evaluate_config(rnd: int):
    """Return evaluation configuration dict for each round.
    Perform five local evaluation steps on each client (i.e., use five
    batches) during rounds one to three, then increase to ten local
    evaluation steps.
    """
    val_steps = 5 if rnd < 4 else 10
    return {"val_steps": val_steps}


if __name__ == "__main__":
    main()