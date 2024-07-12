import tensorflow as tf
import logging
from keras.src.models import Sequential
from keras.src.layers import Dense
from visualize import *


logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.autograph.set_verbosity(0)
np.set_printoptions(precision=2)
tf.random.set_seed(1123)


def recognize(images, labels, out_dim):
    model = Sequential(
        [Dense(units=(2 << i), activation='relu') for i in range(8) if (2 << i) > out_dim] + [Dense(units=out_dim)]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    model.fit(
        images,
        labels,
        epochs=25
    )
    
    return model


if __name__ == '__main__':
    visualize_random_64(
        data='digits',
        model=recognize(
            images=extract_images("emnist-digits-train-images-idx3-ubyte"),
            labels=extract_labels(
                filename="emnist-digits-train-labels-idx1-ubyte",
                mapping=None
            ),
            out_dim=len(extract_mapping("emnist-digits-mapping.txt"))
        )
    )

    visualize_random_64(
        data='letters',
        model=recognize(
            images=extract_images("emnist-letters-train-images-idx3-ubyte"),
            labels=extract_labels(
                filename="emnist-letters-train-labels-idx1-ubyte",
                mapping=None
            ),
            out_dim=len(extract_mapping("emnist-letters-mapping.txt"))
        )
    )
