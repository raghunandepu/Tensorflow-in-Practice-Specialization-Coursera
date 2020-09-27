"""
Exercise 2
In the course you learned how to do classificaiton using Fashion MNIST, a data set containing items of clothing.
There's another, similar dataset called MNIST which has items of handwriting -- the digits 0 through 9.

Write an MNIST classifier that trains to 99% accuracy or above, and does it without a fixed number of epochs -- i.e.
you should stop training once you reach that level of accuracy.

Some notes:

It should succeed in less than 10 epochs, so it is okay to change epochs= to 10, but nothing larger
When it reaches 99% or greater it should print out the string "Reached 99% accuracy so cancelling training!"
If you add any additional variables, make sure you use the same names as the ones used in the class
"""

import tensorflow as tf
from tensorflow import keras
from os import path, getcwd, chdir

path = f"{getcwd()}/mnist.npz"


def train_mnist():
    class MyCallBack(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            if(logs.get('accuracy')>0.99):
                print("\nReached 99% accuracy so cancelling training!")
                self.model.stop_training = True
    callbacks = MyCallBack()

    mnist = tf.keras.datasets.mnist
    (X_train, y_train), (X_test, y_test) = mnist.load_data(path=path)

    # Normalizing the dataset
    X_train = X_train / 255.0
    X_test = X_test / 255.0

    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation="relu"),
        tf.keras.layers.Dense(10, activation="softmax")
    ])

    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])

    history = model.fit(X_train, y_train, epochs=10, callbacks = [callbacks])
    return history.epoch, history.history['accuracy'][-1]

print(train_mnist())