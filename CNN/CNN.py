# Annars finns U-net https://github.com/zhixuhao/unet

# https://pythonprogramming.net/introduction-deep-learning-python-tensorflow-keras/
import tensorflow.keras as keras
import tensorflow as tf


# Import and preprocess data
mnist = tf.keras.datasets.mnist
#(x_train, y_train),(x_test, y_test) = mnist.load_data()

x_test, y_test = dataset.read_test_data()
x_train, y_train = dataset.next_train_batch()


#Vizualise data
import matplotlib.pyplot as plt

plt.imshow(x_train[0],cmap=plt.cm.binary)
plt.show()
print(y_train[0])


# Normalize
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)


# Build model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten()) # Flattens image to one column
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu)) # Fully connected
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax)) # Final output layer


# Optimize and fit
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(x_train, y_train, epochs=3)

val_loss, val_acc = model.evaluate(x_test, y_test)
print(val_loss)
print(val_acc)


# Output

import numpy as np

predictions = model.predict(x_test)

print(np.argmax(predictions[0]))

plt.imshow(x_test[0],cmap=plt.cm.binary)
plt.show()
