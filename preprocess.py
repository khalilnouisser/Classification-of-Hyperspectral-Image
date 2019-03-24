from random import shuffle

import numpy as np
from scipy.io import loadmat

from display_utils import displayClassTable
from preprocess_utils import normalize, create_patch, one_hot_encoding

PATCH_SIZE = 5


def prepareData():
    filename = './data/PaviaU.mat'
    image_dict = loadmat(filename)
    img = np.array(image_dict['paviaU'])

    gt_fileName = './data/PaviaU_gt.mat'
    gt_image_dict = loadmat(gt_fileName)
    img_labels = np.array(gt_image_dict['paviaU_gt'])

    # Etape 0 TODO : name
    image_width = img.shape[0]
    image_height = img.shape[1]
    image_layers = img.shape[2]
    nb_classes = np.max(img_labels)

    # Etape 1 : normalisation de l'image
    img = normalize(img)

    # Etape 2 : preparation de l'image
    img = np.transpose(img, (2, 0, 1))
    padding_width = int((PATCH_SIZE - 1) / 2)
    pixel_means = []  # represente la moyenne d'une pixel

    new_image = []

    for i in range(image_layers):
        pixel_means.append(np.mean(img[i, :, :]))
        # ajouter padding de 0 dans les bordeurs
        p = np.pad(img[i, :, :], padding_width, 'constant', constant_values=0)
        new_image.append(p)

    new_image = np.array(new_image)

    # Etape 3 : Énumerer les echantilions
    nb_samples = np.zeros(nb_classes)
    classes = []
    for i in range(nb_classes):
        classes.append([])

    for i in range(image_width):
        for j in range(image_height):
            label = img_labels[i, j]
            patch = create_patch(new_image, i, j, PATCH_SIZE)
            if label > 0:
                nb_samples[label - 1] += 1
                classes[label - 1].append(patch)

    displayClassTable(nb_samples)

    # Etape 4 : Remplisage des data
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    train_ratio = 0.2

    for classId, classData in enumerate(classes):
        shuffle(classData)
        train_data_size = int(len(classData) * train_ratio)
        x_train += (classData[:train_data_size])
        y_train += ([classId] * train_data_size)
        x_test += (classData[train_data_size:])
        y_test += ([classId] * (len(classData) - train_data_size))

    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_test = np.array(x_test)
    y_test = np.array(y_test)

    x_trains_alea_indexs = list(range(len(x_train)))
    shuffle(x_trains_alea_indexs)
    x_train = x_train[x_trains_alea_indexs]
    y_train = y_train[x_trains_alea_indexs]

    y_trains_alea_indexs = list(range(len(x_test)))
    shuffle(x_trains_alea_indexs)
    x_test = x_test[y_trains_alea_indexs]
    y_test = y_test[y_trains_alea_indexs]

    y_train = one_hot_encoding(y_train)
    y_test = one_hot_encoding(y_test)

    x_train = np.transpose(x_train, (0, 2, 3, 1))
    x_test = np.transpose(x_test, (0, 2, 3, 1))

    print("\n+-------------------------------------+")
    print("Resume")
    print('x_train.shape: ' + str(x_train.shape))
    print('y_train.shape: ' + str(y_train.shape))
    print('x_test.shape: ' + str(x_test.shape))
    print('y_test.shape: ' + str(y_test.shape))
    print("+-------------------------------------+")
    print("Etape PreProcessing est termine")

    return x_train, y_train, x_test, y_test
