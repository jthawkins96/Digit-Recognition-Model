# -*- coding: utf-8 -*-
"""
Created on Thu Aug  9 12:57:10 2018

@author: Jack
"""
from tensorflow.examples.tutorials.mnist import input_data
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
import numpy as np
from matplotlib import pyplot as plt

mnist = input_data.read_data_sets('./', one_hot=True)
x, y = mnist.train.images, mnist.train.labels
dev_x, dev_y = mnist.validation.images, mnist.validation.labels
test_x, test_y = mnist.test.images, mnist.test.labels

x = np.array([img.reshape(28,28,1) for img in x])
dev_x = np.array([img.reshape(28,28,1) for img in dev_x])
test_x = np.array([img.reshape(28,28,1) for img in test_x])

input_shape = x[0].shape

model = Sequential()
model.add(Conv2D(input_shape=input_shape, filters=16, kernel_size=(2,2), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(filters=32, kernel_size=(2,2), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(filters=64, kernel_size=(2,2), activation='relu'))
model.add(MaxPooling2D(pool_size=(1,1)))
model.add(Conv2D(filters=128, kernel_size=(2,2), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(units=100, activation="relu"))
model.add(Dense(units=10, activation="softmax"))
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(x, y, epochs=5, batch_size=2048, shuffle=True)
print("\n")

score = model.evaluate(dev_x, dev_y)
print("----Dev Set----\nLoss: " + str(score[0]) + "\nAccuracy: " + str(round(score[1]*100, 3))+"%\n")

score = model.evaluate(test_x, test_y)
print("----Test Set----\nLoss: " + str(score[0]) + "\nAccuracy: " + str(round(score[1]*100, 3))+"%\n")

predictions = model.predict(test_x, 1)
print("Model predicts a: " + str(np.argmax(predictions[1])))
plt.imshow(test_x[1].reshape(28,28), cmap="Greys")


