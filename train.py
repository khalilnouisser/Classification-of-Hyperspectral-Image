from keras.layers import Dense, Conv2D, Flatten, Dropout, Conv3D
from keras.models import Sequential

from preprocess import prepareData


def network(width, height, layers):
    C1 = 3 * layers

    # Define the model
    model = Sequential()
    #model.add(Conv2D(C1, (3, 3), activation='relu', input_shape=input_shape))
    model.add(Conv2D(3 * C1, (3, 3), activation='relu'))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(6 * layers, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(16, activation='softmax'))
    model.summary()
    return model


if __name__ == '__main__':
    train_image, train_labels, test_image, test_labels = prepareData()
    #model = network(train_image.shape[0], train_image.shape[1], train_image.shape[2])
