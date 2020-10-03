"""
Exercise 3
In the videos you looked at how you would improve Fashion MNIST using Convolutions. For your exercise see if you can
improve MNIST to 99.8% accuracy or more using only a single convolutional layer and a single MaxPooling 2D.
You should stop training once the accuracy goes above this amount. It should happen in less than 20 epochs,
so it's ok to hard code the number of epochs for training, but your training must end once it hits the above metric.
If it doesn't, then you'll need to redesign your layers.

When 99.8% accuracy has been hit, you should print out the string "Reached 99.8% accuracy so cancelling training!"
"""

import tensorflow as tf
from tensorflow import keras
from os import path, getcwd, chdir
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

path = f"{getcwd()}/../tmp/mnist.npz"

mnist = tf.keras.datasets.mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data(path=path)

print(X_train.shape)


def train_mnist_conv():
    DESIRED_ACCURACY = 0.998
    class myCallback(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs = {}):
            if(logs.get('accuracy') > DESIRED_ACCURACY):
                print("\nReached 99.8% accuracy so cancelling training!")
                self.model.stop_training = True
    callbacks = myCallback()

    mnist = keras.datasets.mnist
    (X_train, y_train), (X_test, y_test) = mnist.load_data(path=path)

    # Reshaping images for convolutions
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[1], 1)
    X_train = X_train / 255.0

    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[1], 1)
    X_test = X_test / 255.0

    # Creating model
    model = keras.models.Sequential([
        keras.layers.Conv2D(64, (3,3), activation="relu", input_shape=(28,28,1)),
        keras.layers.MaxPooling2D(2,2),
        keras.layers.Conv2D(64, (3, 3), activation="relu"),
        keras.layers.MaxPooling2D(2, 2),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dense(10, activation="sigmoid")
    ])

    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=['accuracy'],)
    history = model.fit(X_train, y_train, epochs = 20,  callbacks=[callbacks])

    return history.epoch, history.history['accuracy'][-1]


_, _ = train_mnist_conv()