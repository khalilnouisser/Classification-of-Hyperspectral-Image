import h5py
import numpy as np


def load_matfile(filename='./data/indian_pines_data.mat'):
    print(filename)
    f = h5py.File(filename, 'r')
    print(f)
    print(f['X_r'].shape)
    if 'pca' in filename:
        X = np.asarray(f['X_r'], dtype='float32')
    else:
        X = np.asarray(f['X'], dtype='float32')
    y = np.asarray(f['labels'], dtype='uint8')
    gt = np.asarray(f['ip_gt'], dtype='uint8')
    f.close()

    X = X.transpose(3, 2, 1, 0)
    y = np.squeeze(y) - 1
    gt = gt.transpose(1, 0)
    return X, y, gt


if __name__ == '__main__':
    X, y, gt = load_matfile(filename='./data/Indian_pines_pca.mat')
