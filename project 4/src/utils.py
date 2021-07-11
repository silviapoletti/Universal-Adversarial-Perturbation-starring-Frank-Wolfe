from lenet5 import *
import numpy as np
import matplotlib.pyplot as plt


def get_data(dim=100, load=True):
    """
    :param dim: images per worker
    :param load:
    :return: data_workers, y, lenet5, right_pred_data, labels,  test_x, test_y.
                Where right_pred_data is the variable of lenet5 correctly classified data.
    """

    # ------------------- DATA preparation -------------------
    _, (test_x, test_y) = load_MNIST()

    path = '../data/lenet5'
    lenet5 = LeNet5(path=path, load=load)

    lab = lenet5.predict(test_x)
    indexes = lab == test_y
    right_pred_data = test_x[indexes]
    right_pred_labels = test_y[indexes]

    print(len(right_pred_labels))  # 9826

    labels_number = 10
    data_per_classes = []
    for label_class in range(0, labels_number):
        data_per_classes.append(right_pred_data[right_pred_labels == label_class][:dim])

    data_per_classes = np.array(data_per_classes)
    data_workers = []

    step = dim//labels_number
    print("step", step)
    print(data_per_classes.shape)
    for offset in range(0, dim, step):
        image_worker = []
        for c in range(0, labels_number):
            image_worker.extend(data_per_classes[c, offset:offset+step, :, :, :])
        data_workers.append(image_worker)

    data_workers = np.array(data_workers)
    print(data_workers.shape)  # now all 10 workers have dim images, step for each class.

    y_workers = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    y_workers = np.repeat(y_workers, step)

    return data_workers, y_workers, lenet5, right_pred_data, right_pred_labels,  test_x, test_y

def get_image_perturbation(perturbation, title, axis):
    """
    :param perturbation: a  numpy perturbation to plot
    :param title: title to give to the image genarated
    :return: the plot of the axis and the image
    """
    img = axis.imshow(perturbation.reshape((28, 28)))
    axis.set_title(title)
    plt.colorbar(img, ax=axis, fraction=0.03, pad=0.05)
    return axis, img

def plot_perturbation(perturbation, title, file_path=None, figsize=(5,5)):
    fig, ax = plt.subplots(figsize=figsize)
    ax, img = get_image_perturbation(perturbation, title, ax)
    if file_path is not None:
        plt.savefig(file_path, bbox_inches="tight")
    plt.show()

def get_image_perturbed(perturbation, image_test, title, axis):
    image = image_test.reshape(28, 28)
    img_noise = image + perturbation.reshape((28, 28))
    img_noise = np.clip(img_noise, 0., 1.)
    img = axis.imshow(img_noise, cmap='Greys')
    axis.set_title(title)
    plt.colorbar(img, ax=axis, fraction=0.03, pad=0.05)
    return axis, img

def plot_perturbed_img(perturbation, image_test, file_path=None, figsize=(5,5)):
    """

    :param perturbation: a numpy perturbation
    :param image_test: a numpy image
    :param file_path:
    :return:
    """
    image = image_test.reshape(28, 28)
    img_noise = image + perturbation.reshape((28, 28))
    img_noise = np.clip(img_noise, 0., 1.)
    fig, ax = plt.subplots(1, 2, figsize=figsize)
    a = ax[0].imshow(image, cmap='Greys')
    b = ax[1].imshow(img_noise, cmap='Greys')
    ax[0].set_title("Real image")
    ax[1].set_title("Perturbed image")
    fig.colorbar(a, ax=ax[0], fraction=0.03, pad=0.05)
    fig.colorbar(b, ax=ax[1], fraction=0.03, pad=0.05)
    fig.tight_layout(pad=0.3)
    if file_path is not None:
        plt.savefig(file_path, bbox_inches="tight")
    plt.show()

def predict_single_img_perturbation(lenet5, image, delta):
    """
    :param image: Single image to predict
    :param delta: perturbation
    :return: predicted class
    """
    image_noise = image + delta.reshape(28, 28, 1)
    image_noise = np.clip(image_noise, 0., 1.)
    return lenet5.predict(np.array([image_noise]))[0]

def predict_images_perturbation(lenet5, images, delta):
    deltas = np.tile(delta, images.shape[0])
    images_noise = images + deltas.reshape(images.shape[0], 28, 28, 1)
    images_noise = np.clip(images_noise, 0., 1.)
    return lenet5.predict(images_noise)

def evaluate_perturbed_images(lenet5, images, labels, delta, verbose=0):
    deltas = np.tile(delta, images.shape[0])
    images_noise = images + deltas.reshape(images.shape[0], 28, 28, 1)
    images_noise = np.clip(images_noise, 0., 1.)
    return lenet5.model.evaluate(images_noise, labels, verbose=verbose)

def get_distributed_best_delta(lenet5, images, labels, delta_workers, verbose=0):
    best_delta = None
    loss = -1
    worker_idx = 0
    for idx in range(delta_workers.shape[0]):
        delta = delta_workers[idx]
        new_loss = evaluate_perturbed_images(lenet5, images, labels, delta, verbose)[0]
        if new_loss > loss:
            loss = new_loss
            best_delta = delta
            worker_idx = idx

    return best_delta, worker_idx

def plot_loss(loss_history, m):
    plt.figure(figsize=(10,6))
    plt.plot(m, loss_history[:, 0])
    plt.title('loss')
    plt.show()

def plot_accuracy(loss_history, m):
    plt.figure(figsize=(10,6))
    plt.plot(m, loss_history[:, 1])
    plt.title('accuracy')
    plt.show()

