# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 10:01:26 2020

@author: mhauck
"""

import numpy as np

from keras.preprocessing.image import ImageDataGenerator
from keras.datasets import cifar10
import keras.models as models

from pathlib import Path

from PIL import Image
from PIL import ImageDraw

import matplotlib.pyplot as plt

# DATA PREPARATION #


picture_path = Path(r'')
model_path = Path(r'')

img = Image.open(picture_path)
#img = img.resize((int(img.size[0] / 1), int(img.size[1] / 1)), resample = Image.BICUBIC)

#img_np = np.asarray(img.crop((750,750,750+576,750+576)).resize((32, 32), resample = Image.BICUBIC)).astype(np.float32) / 255.
#img_np = img_np.reshape(1, 32, 32, 3)

#gcd = np.gcd(img.size[0],img.size[1])
gcd = 400

if gcd < 32:
    
    print('WARNING! Common divisor of picture dimensions is less than 32!')

pix_step = 20
    
x_dim = img.size[0]
y_dim = img.size[1]

x_dim_boxed = x_dim - gcd
y_dim_boxed = y_dim - gcd

(X_train, y_train), (X_test, y_test) = cifar10.load_data()

y_train_car = y_train == 1
y_test_car = y_test == 1

X_train = X_train.astype(np.float32) / 255.
X_test = X_test.astype(np.float32) / 255.

# Car is class 1 in training data

# LOAD_MODEL #

model = models.load_model(model_path)

# IMAGE DATA GENERATOR FOR PREDICTION #

gen_pred = ImageDataGenerator(featurewise_center = True,
                              featurewise_std_normalization = True)

gen_pred.fit(X_train)

# MODEL PREDICTION #

img_np_arr = np.zeros((int(np.ceil(x_dim_boxed/pix_step) * np.ceil(y_dim_boxed/pix_step)), 32, 32, 3))
ind_arr = np.zeros((int(np.ceil(x_dim_boxed/pix_step) * np.ceil(y_dim_boxed/pix_step)), 2))

k = 0

for i in range(0, x_dim_boxed, pix_step):
    for j in range(0, y_dim_boxed, pix_step):

        ind_arr[k,0] = i
        ind_arr[k,1] = j
        
        img_np_arr[k,:,:,:] = np.asarray(img.crop((i,j,i+gcd,j+gcd)).resize((32, 32), resample = Image.BICUBIC)).astype(np.float32).reshape(32, 32, 3) / 255.
        
        k += 1
       # prob_pred[i,j] = model.predict(gen_pred.flow(img_np))

prob_pred = model.predict(gen_pred.flow(img_np_arr, batch_size = 1, shuffle = False))
prob_pred = (prob_pred).reshape(-1)

draw = ImageDraw.Draw(img)

#pts = [(max_i, max_j), (max_i + gcd, max_j + gcd)]


wh = np.argwhere((prob_pred[:] >= 0.90) & (prob_pred[:] <= 1.))
wh = np.append(wh, 1E9)
wh_group = []

for i in range(1,len(wh)):
    
    wh_group.append(wh[i-1])
    
    if np.abs(wh[i-1] - wh[i]) > gcd / 2.:
        
        wh_group = np.asarray(list(map(int, wh_group)))
        pts = [np.min(ind_arr[wh_group[:],0]), np.min(ind_arr[wh_group[:],1]), np.max(ind_arr[wh_group[:],0]) + gcd, np.max(ind_arr[wh_group[:],1]) + gcd]
        wh_group = []

        draw.rectangle(pts, outline = 'red', width = 10)
        plt.imshow(img)

plt.show()