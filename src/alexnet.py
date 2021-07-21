import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, models, losses


class AlexNet:

    def __init__(self, path, load=True):
        if load:
            self.load_model(path)
        else:
            self.create_model(path)

    def create_model(self, path):
        (x_train, y_train), _ = datasets.mnist.load_data()
        x_train = tf.pad(x_train, [[0, 0], [2, 2], [2, 2]]) / 255
        x_train = tf.expand_dims(x_train, axis=3, name=None)
        x_train = tf.repeat(x_train, 3, axis=3)
        x_val = x_train[-2000:, :, :, :]
        y_val = y_train[-2000:]
        x_train = x_train[:-2000, :, :, :]
        y_train = y_train[:-2000]

        model = models.Sequential()
        model.add(layers.experimental.preprocessing.Resizing(224, 224, interpolation="bilinear",
                                                             input_shape=x_train.shape[1:]))
        model.add(layers.Conv2D(96, 11, strides=4, padding='same'))
        model.add(layers.Lambda(tf.nn.local_response_normalization))
        model.add(layers.Activation('relu'))
        model.add(layers.MaxPooling2D(3, strides=2))
        model.add(layers.Conv2D(256, 5, strides=4, padding='same'))
        model.add(layers.Lambda(tf.nn.local_response_normalization))
        model.add(layers.Activation('relu'))
        model.add(layers.MaxPooling2D(3, strides=2))
        model.add(layers.Conv2D(384, 3, strides=4, padding='same'))
        model.add(layers.Activation('relu'))
        model.add(layers.Conv2D(384, 3, strides=4, padding='same'))
        model.add(layers.Activation('relu'))
        model.add(layers.Conv2D(256, 3, strides=4, padding='same'))
        model.add(layers.Activation('relu'))
        model.add(layers.Flatten())
        model.add(layers.Dense(4096, activation='relu'))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(4096, activation='relu'))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(10, activation='softmax'))

        model.compile(optimizer='adam', loss=losses.sparse_categorical_crossentropy, metrics=['accuracy'])
        history = model.fit(x_train, y_train, batch_size=64, epochs=5, validation_data=(x_val, y_val))

        self.model = model

        model.save(path)

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
        return -self[0]
