from keras.layers import Dense, Conv2D, Flatten
from keras.models import Sequential

from prepossessing import prepareData


def network(width, height, layers):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        Flatten(),
        Dense(2, activation='softmax')
    ])

    model.summary()
    return model


if __name__ == '__main__':
    train_image, train_labels, test_image, test_labels = prepareData()
    model = network(train_image.shape[0], train_image.shape[1], train_image.shape[2])
