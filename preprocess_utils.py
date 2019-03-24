import numpy as np
from keras.utils import to_categorical
from numpy import array


# TODO : ajouter description

def one_hot_encoding(data):
    data = array(data)
    encoded = to_categorical(data)
    return encoded


"""
Methode qui permet de normaliser une image
"""


# TODO : ajouter description

def normalize(img):
    min = np.min(img)
    max = np.max(img)
    img = img.astype(np.float32)
    img = img - min
    img = img / max
    return img


def create_patch(input_mat, height_index, width_index, PATCH_SIZE):
    # Input:
    # Étant donné la position d'index (x, y) de la dimension spatiale de l'image hyperspectrale

    # Output:
    # un cube de données avec une taille de patch Size (24 voisins), avec une étiquette basée sur le pixel central

    height_slice = slice(height_index, height_index + PATCH_SIZE)
    width_slice = slice(width_index, width_index + PATCH_SIZE)

    patch = input_mat[:, height_slice, width_slice]
    # mean_normalized_patch = []
    # for i in range(patch.shape[0]):
    #    mean_normalized_patch.append(patch[i] - MEAN_ARRAY[i])

    return np.array(patch).astype(np.float32)
