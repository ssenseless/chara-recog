import os
import numpy as np


def extract_images(filename):
    path = os.path.abspath(os.getcwd())
    path += f"\\data\\gzip\\{filename}"

    map = np.memmap(path, mode='r', dtype='uint8')[16:]

    if len(map) % 784 != 0:
        print("something wrong reading image binaries")
    else:
        images = [map[(784 * i):(784 * (i + 1))] for i in range(int(len(map) / 784))]

    return np.array(images)


def extract_labels(filename, mapping=None):
    path = os.path.abspath(os.getcwd())
    path += f"\\data\\gzip\\{filename}"

    labels = []
    map = np.memmap(path, mode='r', dtype='uint8')[8:]
    if mapping is not None:
        for label in map:
            label = chr(mapping[label])
            labels.append(label)
    else:
        for label in map:
            labels.append(label)

    return np.array(labels)


def extract_mapping(filename):
    path = os.path.abspath(os.getcwd())
    path += f"\\data\\gzip\\{filename}"

    mapping = {}
    with open(path, 'r') as file:
        for line in file:
            mappings = line[:-1].split(" ")
            for i in range(len(mappings) - 1):
                mapping.update({int(mappings[i]): int(mappings[len(mappings) - 1])})

    return mapping


if __name__ == "__main__":
    test_i = extract_images("emnist-bymerge-train-images-idx3-ubyte")
    test_l = extract_labels(filename="emnist-bymerge-train-labels-idx1-ubyte")
    test_m = extract_mapping("emnist-letters-mapping.txt")

    print(type(test_i))
    print(type(test_l))
    print(type(test_m))
