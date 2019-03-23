from keras.utils import to_categorical
from numpy import array


def one_hot_encoding(data):
    data = array(data)
    encoded = to_categorical(data)
    return encoded
