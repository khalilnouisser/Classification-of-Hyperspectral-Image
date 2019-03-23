import matplotlib.pyplot as plt


def display_image_with_multiple_layers(image):
    for i in range(0, image.shape[2]):
        img = image[:, :, i]
        plt.imshow(img, alpha=.3)
    plt.show()


def display_data_diff(image, train_image, test_image):
    fig = plt.figure(figsize=(10, 3))
    ax = []

    ax.append(fig.add_subplot(1, 3, 1))
    ax[-1].set_title("data")
    plt.imshow(image[:, :, 0])

    ax.append(fig.add_subplot(1, 3, 2))
    ax[-1].set_title("train")
    plt.imshow(train_image[:, :, 0])

    ax.append(fig.add_subplot(1, 3, 3))
    ax[-1].set_title("test")
    plt.imshow(test_image[:, :, 0])

    plt.show()


def display_one_dim_image(img):
    plt.imshow(img)
    plt.show()
