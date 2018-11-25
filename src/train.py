#!/usr/bin/env python
# -*- coding: utf-8 -*-
from PIL import Image
import matplotlib as plt
import numpy as np
import os
import timeit
import keras
import sys
from keras.models import Sequential
from keras.models import save_model
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from os.path import basename
from sklearn.model_selection import train_test_split

#paths to find train and test data

epochs = sys.argv[1]
path = sys.argv[2]

train_Images_1_dir = '/home/dmitriy/Documents/Dataset/Images_0/'
train_Images_2_dir = '/home/dmitriy/Documents/Dataset/Images_1/'
train_Images_3_dir = '/home/dmitriy/Documents/Dataset/Images_2/'
train_Images_4_dir = '/home/dmitriy/Documents/Dataset/Images_3/'

train_Images_1_sample = len(os.listdir(train_Images_1_dir))
train_Images_2_sample = len(os.listdir(train_Images_2_dir))
train_Images_3_sample = len(os.listdir(train_Images_3_dir))
train_Images_4_sample = len(os.listdir(train_Images_4_dir))

images_to_analyze=np.zeros([train_Images_1_sample+train_Images_2_sample+train_Images_3_sample+train_Images_4_sample,100,100,3])

labels_to_analyze = np.array([])


for index,filename in enumerate(os.listdir(train_Images_1_dir)):
    img = Image.open(train_Images_1_dir+filename)
    img = img.resize((100,100),Image.ANTIALIAS)
    im = np.array(img)
    images_to_analyze[index, :, :, :] = im
    labels_to_analyze = np.append(labels_to_analyze, 0)

for index2,filename in enumerate(os.listdir(train_Images_2_dir)):
    img = Image.open(train_Images_2_dir+filename)
    img = img.resize((100,100),Image.ANTIALIAS)
    im = np.array(img)
    images_to_analyze[index+index2, :, :, :] = im
    labels_to_analyze = np.append(labels_to_analyze, 1)

for index3,filename in enumerate(os.listdir(train_Images_3_dir)):
    img = Image.open(train_Images_3_dir+filename)
    img = img.resize((100,100),Image.ANTIALIAS)
    im = np.array(img)
    images_to_analyze[index+index2+index3, :, :, :] = im
    labels_to_analyze = np.append(labels_to_analyze, 2)

for index4,filename in enumerate(os.listdir(train_Images_4_dir)):
    img = Image.open(train_Images_4_dir+filename)
    img = img.resize((100,100),Image.ANTIALIAS)
    im = np.array(img)
    images_to_analyze[index+index2+index3+index4, :, :, :] = im
    labels_to_analyze = np.append(labels_to_analyze, 3)

images_to_analyze = images_to_analyze/255

X_train, X_test, y_train, y_test = train_test_split(images_to_analyze, labels_to_analyze, test_size=0.1)
#train_y_data_set=np.array([])

y_train = keras.utils.to_categorical(y_train, num_classes = 4)
y_test = keras.utils.to_categorical(y_test, num_classes = 4)

#print(train_y_data_set.shape)
#print("shape of train label:"+str(train_y_data_set.shape))
print (X_train.shape)
print(y_train.shape)

model = Sequential()
model.add(Conv2D(8,(3,3),input_shape=(100,100,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(12,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(16,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(units=128,activation='relu'))
model.add(Dense(units=4,activation='softmax'))
model.summary()

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

model.fit(X_train,y_train,epochs=int(epochs))

score = model.evaluate(X_test, y_test)

#model.save("path/model.h5")
save_model(
    model,
    path,
    overwrite=True,
    include_optimizer=True
)