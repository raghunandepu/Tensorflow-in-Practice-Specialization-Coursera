#Improving Computer Vision Accuracy using Convolutions

import tensorflow as tf
from tensorflow import keras

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

mnist = keras.datasets.fashion_mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()



"""model = keras.models.Sequential([
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(10, activation="softmax")

])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=['accuracy'])
model.fit(X_train, y_train, epochs=5)

test_loss = model.evaluate(X_test, y_test)
print(test_loss)"""

"""Your accuracy is probably about 89% on training and 87% on validation...not bad...But how do you make that even 
better? One way is to use something called Convolutions."""

print(X_train.shape) # (60000, 28, 28)
print(X_test.shape) # (10000, 28, 28)

# Reshaping for convolutions
X_train = X_train.reshape(60000, 28, 28, 1)
X_test = X_test.reshape(10000, 28, 28, 1)

X_train = X_train / 255.0
X_test = X_test / 255.0

model = keras.models.Sequential([
    keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(28, 28, 1)),
    keras.layers.MaxPooling2D(2,2),
    keras.layers.Conv2D(64, (3,3), activation='relu'),
    keras.layers.MaxPooling2D(2,2),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='sigmoid')
])

model.compile(optimizer="adam", loss='sparse_categorical_crossentropy', metrics=['accuracy'])
print(model.summary())

model.fit(X_train, y_train, epochs=5)
test_loss = model.evaluate(X_test, y_test)

print(test_loss)

"""
It's likely gone up to about 93% on the training data and 91% on the validation data.

That's significant, and a step in the right direction!

Try running it for more epochs -- say about 20, and explore the results! But while the results might seem really good, 
the validation results may actually go down, due to something called 'overfitting'.
"""