import pandas as pd
import numpy as np
import tensorflow as tf


def format_data(file_path):
    data = pd.read_csv(file_path, index_col=0)
    labels = data["Survived"]
    labels = np.expand_dims(labels, 1)
    features = data.drop(["Survived"], 1,)
    return features, labels


def create_layer(layer, weights, biases, activation_function = tf.nn.relu):
    z = tf.matmul(layer, weights) + biases

    if activation_function is not None:
        return activation_function(z)

    return z