from itertools import product

import numpy as np
from scipy.io import loadmat

from display_utils import display_data_diff, display_one_dim_image
from prepossessing_utils import one_hot_encoding


def prepareData():
    filename = './data/PaviaU.mat'
    image_dict = loadmat(filename)
    img = image_dict['paviaU']

    gt_fileName = './data/PaviaU_gt.mat'
    gt_image_dict = loadmat(gt_fileName)
    img_labels = gt_image_dict['paviaU_gt']

    display_one_dim_image(img_labels)

    possible_pixels = np.array(list(product(np.arange(img.shape[0]), np.arange(img.shape[1]))))
    nb_alea_pixels = np.int(len(possible_pixels) * 0.2)
    alea_pixels_index = np.random.choice(np.arange(len(possible_pixels)), nb_alea_pixels)
    alea_pixels = possible_pixels[alea_pixels_index]

    train_image = np.zeros(img.shape, dtype=np.float64)
    train_labels = np.zeros(img_labels.shape, dtype=np.int)

    for (key, value) in alea_pixels:
        train_image[key, value, :] = img[key, value, :]

    train_labels[alea_pixels] = img_labels[alea_pixels]

    test_image = img - train_image
    test_labels = img_labels - train_labels

    display_data_diff(img, train_image, test_image)

    train_labels = one_hot_encoding(train_labels)
    test_labels = one_hot_encoding(test_labels)

    return train_image, train_labels, test_image, test_labels
