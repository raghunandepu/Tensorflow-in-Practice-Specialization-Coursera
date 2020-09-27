import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Fashion MNIST example

mnist = tf.keras.datasets.fashion_mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()

print(X_train.shape)
print(y_train.shape)

"""# Lets print a training image
np.set_printoptions(linewidth=200)
plt.imshow(X_train[0])
print(X_train[0])
print(y_train[0])
"""

# Normalizing the images
X_train = X_train / 255.0
X_test = X_test / 255.0

# Modelling
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=[28, 28]),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax")
])

model.compile(optimizer="Adam",
              loss="sparse_categorical_crossentropy",
              metrics=['accuracy'])


class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if(logs.get('accuracy') > 0.95):
            print("\nReached required accuracy so cancelling training!")
            self.model.stop_training  = True

callbacks = myCallback()

model.fit(X_train, y_train, epochs=100, callbacks=[callbacks])

# Returns loss value and metric
print(model.evaluate(X_test, y_test))

# Observation: This model return 87.84% accuracy.

# Testing
classifications = model.predict(X_test)
print(classifications[0])
print(y_test[0])

