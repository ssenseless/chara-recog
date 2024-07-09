import numpy as np
import matplotlib.pyplot as plt
from extract import extract_images, extract_labels, extract_mapping


def visualize():
    images = extract_images(filename="emnist-bymerge-train-images-idx3-ubyte")
    labels = extract_labels(filename="emnist-bymerge-train-labels-idx1-ubyte", mapping=extract_mapping("emnist-bymerge-mapping.txt"))

    for i in range(len(images)):
        plt.title(f"Label: {labels[i]}")
        plt.imshow(images[i], cmap='gray')
        plt.show()


if __name__ == "__main__":
    visualize()
