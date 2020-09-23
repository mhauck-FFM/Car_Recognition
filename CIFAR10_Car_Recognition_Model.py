# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 10:01:26 2020

@author: mhauck
"""

import numpy as np

from keras.preprocessing.image import ImageDataGenerator
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.optimizers import RMSprop
import keras.backend as K
import keras.models as models

from pathlib import Path

# DATA PREPARATION #

model_path = Path(r'')

(X_train, y_train), (X_test, y_test) = cifar10.load_data()

y_train_car = y_train == 1
y_test_car = y_test == 1

X_train = X_train.astype(np.float32) / 255.
X_test = X_test.astype(np.float32) / 255.

# Car is class 1 in training data

# MODEL CORE #

model = Sequential()
model.add(Conv2D(32, kernel_size = (3, 3), activation = 'relu', input_shape = (32, 32, 3), padding = 'same'))
model.add(Conv2D(32, kernel_size = (3, 3), activation = 'relu', padding = 'same'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, kernel_size = (3, 3), activation = 'relu', padding = 'same'))
model.add(Conv2D(64, kernel_size = (3, 3), activation = 'relu', padding = 'same'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, kernel_size = (3, 3), activation = 'relu', padding = 'same'))
model.add(Conv2D(128, kernel_size = (3, 3), activation = 'relu', padding = 'same'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, activation = 'relu'))
model.add(Dense(128, activation = 'relu'))
model.add(Dense(1, activation = 'sigmoid'))

model.compile(optimizer = RMSprop(lr = 0.0001),
              loss = 'binary_crossentropy',
              metrics = ['accuracy'])

print(model.summary())

# IMAGE DATA GENERATOR #

gen = ImageDataGenerator(width_shift_range = 3, 
                         height_shift_range = 3, 
                         zoom_range = 0.1, 
                         horizontal_flip = True,
                         featurewise_center = True,
                         featurewise_std_normalization = True)

gen.fit(X_train)

gen_pred = ImageDataGenerator(featurewise_center = True,
                              featurewise_std_normalization = True)

gen_pred.fit(X_train)

'''
Huge downside of Keras: You can't save the normalization and centering statistics for prediction

Theoretically you would need the same generator as in gen, but without the shifting and rotating
What you could do is the following:

    featurewise_center: X_test -= gen.mean
    
    featurewise_std_norm: X_test /= (gen.std + K.epsilon())    

Or you create gen_pred, where only normalization and centering are included, and train it on training data
'''

# MODEL FITTING #

model.fit(gen.flow(X_train,
          y_train_car,
          batch_size = 128),
          epochs = 50,
          steps_per_epoch = np.ceil(len(X_train) / 128))

print(model.evaluate(gen_pred.flow(X_test, y_test_car, batch_size = 128)))
#print(model.evaluate(X_test, y_test_car))

# MODEL SAVING #

model.save(model_path)
