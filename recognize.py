import numpy as np
import tensorflow as tf
import logging
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.activations import linear, relu, sigmoid
from extract import *
from visualize import *


logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.autograph.set_verbosity(0)
np.set_printoptions(precision=2)
tf.random.set_seed(1123)


def recognize(images, labels):
    model = Sequential(
        [
            Dense(units=512, activation='relu'),
            Dense(units=128, activation='relu'),
            Dense(units=10)
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    model.fit(
        images,
        labels,
        epochs=10
    )
    
    return model


if __name__ == '__main__':
    visualize_random_64(
        data='digits',
        model=recognize(
            extract_images("emnist-digits-train-images-idx3-ubyte"),
            extract_labels(
                filename="emnist-digits-train-labels-idx1-ubyte",
                mapping=None
            )
        )
    )
    