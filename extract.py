import os
import numpy as np


def extract_images(filename):
    path = os.path.abspath(os.getcwd())
    path += f"\\data\\gzip\\{filename}"

    images = []
    map = np.memmap(path, mode='c', dtype='uint8')[16:]
    while map is not None:
        if len(map) == 784:
            images.append(map[:784].reshape(28, 28))
            map = None
        else:
            images.append(map[:784].reshape(28, 28).T)
            map = map[784:]

    return images


def extract_labels(filename, mapping=None):
    path = os.path.abspath(os.getcwd())
    path += f"\\data\\gzip\\{filename}"

    labels = []
    map = np.memmap(path, mode='c', dtype='uint8')[8:]
    if mapping is not None:
        for label in map:
            label = chr(mapping[label])
            labels.append(label)
    else:
        for label in map:
            labels.append(label)

    return labels


def extract_mapping(filename):
    path = os.path.abspath(os.getcwd())
    path += f"\\data\\gzip\\{filename}"

    mapping = {}
    with open(path, 'r') as file:
        for line in file:
            mappings = line.split(" ")
            mappings[1] = mappings[1].replace("\n", "")
            mapping.update({int(mappings[0]): int(mappings[1])})

    return mapping

if __name__ == "__main__":
    test_i = extract_images("emnist-bymerge-train-images-idx3-ubyte")
    test_l = extract_labels(filename="emnist-bymerge-train-labels-idx1-ubyte", mapping=extract_mapping("emnist-bymerge-mapping.txt"))

    print(len(test_i))
    print(len(test_l))
