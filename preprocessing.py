#preprocessing
import numpy as np

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

ds1 = unpickle('Datasets/cifar-10-batches-py/data_batch_1')
ds2 = unpickle('Datasets/cifar-10-batches-py/data_batch_2')
print("ds1")
print(ds1[b'labels'])

print(ds1[b'data'])
