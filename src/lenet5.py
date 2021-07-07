import tensorflow as tf
from tensorflow import keras
import numpy as np


class LeNet5:

    def __init__(self, path, load=True):
        if load:
            self.load_model(path)
        else:
            self.create_model(path)

    def create_model(self, path):
        (train_x, train_y), _ = load_MNIST()
        val_x = train_x[:5000]
        val_y = train_y[:5000]
        train_x = train_x[5000:]
        train_y = train_y[5000:]

        lenet_5_model = keras.models.Sequential([
            keras.layers.Conv2D(6, kernel_size=5, strides=1, activation='tanh', input_shape=train_x[0].shape,
                                padding='same'),  # C1
            keras.layers.AveragePooling2D(),  # S2
            keras.layers.Conv2D(16, kernel_size=5, strides=1, activation='tanh', padding='valid'),  # C3
            keras.layers.AveragePooling2D(),  # S4
            keras.layers.Flatten(),  # Flatten
            keras.layers.Dense(120, activation='tanh'),  # C5
            keras.layers.Dense(84, activation='tanh'),  # F6
            keras.layers.Dense(10, activation='softmax')  # Output layer
        ])

        lenet_5_model.compile(optimizer='adam', loss=keras.losses.sparse_categorical_crossentropy, metrics=['accuracy'])
        lenet_5_model.fit(train_x, train_y, epochs=5, validation_data=(val_x, val_y))

        # lenet_5_model.evaluate(test_x, test_y)

        self.model = lenet_5_model

        lenet_5_model.save(path)

    def load_model(self, path):
        self.model = keras.models.load_model(path)
        return self.model

    def predict(self, x):
        """
        :param x: it should be an array of 1 or more elements
        :return: predicted classes
        """
        return self.model.predict_classes(x)

    def negative_loss(self, x, y, verbose=0):
        """
        :param x: it should be an array of 1 or more elements
        :param y: it should be an array of 1 or more elements
        :param verbose:
        :return: negative loss value
        """
        return -self.model.evaluate(x, y, verbose=verbose)[0]


def load_MNIST():
    (train_x, train_y), (test_x, test_y) = keras.datasets.mnist.load_data()
    train_x = train_x / 255.0
    test_x = test_x / 255.0
    train_x = tf.expand_dims(train_x, 3)
    test_x = tf.expand_dims(test_x, 3)

    return (train_x, train_y), (test_x, test_y)