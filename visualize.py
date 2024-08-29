import keras
import numpy as np
import matplotlib.pyplot as plt
from extract import *


def visualize_random_64(data: str, model) -> None:
    map = extract_mapping(mapping_map[data])
    images = extract_images(filename=image_map[data])
    labels = extract_labels(filename=label_map[data],
                            mapping=map)

    fig, axes = plt.subplots(8, 8, figsize=(7, 7))
    fig.tight_layout(pad=0.1, rect=(0, 0.03, 1, 0.91))

    for axis in axes.flat:
        rand = np.random.randint(images.shape[0])
        image = images[rand]
        prediction = np.argmax(model.predict(image.reshape(1, 784)))

        axis.imshow(image.reshape(28, 28).T, cmap='gray')
        axis.set_title(f"true: {labels[rand]}\npred: {chr(map[prediction])}", fontsize=8)
        axis.set_axis_off()
        fig.suptitle('')
    plt.show()


image_map = {
    "balanced": "emnist-balanced-test-images-idx3-ubyte",
    "byclass": "emnist-byclass-test-images-idx3-ubyte",
    "bymerge": "emnist-bymerge-test-images-idx3-ubyte",
    "digits": "emnist-digits-test-images-idx3-ubyte",
    "letters": "emnist-letters-test-images-idx3-ubyte",
    "mnist": "emnist-mnist-test-images-idx3-ubyte"
}

label_map = {
    "balanced": "emnist-balanced-test-labels-idx1-ubyte",
    "byclass": "emnist-byclass-test-labels-idx1-ubyte",
    "bymerge": "emnist-bymerge-test-labels-idx1-ubyte",
    "digits": "emnist-digits-test-labels-idx1-ubyte",
    "letters": "emnist-letters-test-labels-idx1-ubyte",
    "mnist": "emnist-mnist-test-labels-idx1-ubyte"
}

mapping_map = {
    "balanced": "emnist-balanced-mapping.txt",
    "byclass": "emnist-byclass-mapping.txt",
    "bymerge": "emnist-bymerge-mapping.txt",
    "digits": "emnist-digits-mapping.txt",
    "letters": "emnist-letters-mapping.txt",
    "mnist": "emnist-mnist-mapping.txt"
}

if __name__ == "__main__":
    print("ok")
