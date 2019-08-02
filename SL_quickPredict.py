'''
MIT License

Copyright (c) 2019 Rajdeep Bandopadhyay

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''
import string
import numpy as np
import os
import cv2 as cv2
import matplotlib.pyplot as plt
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import RMSprop


size = 200
path_to_pic = "photo.jpg"
imgax4 = cv2.imread(os.path.join(path_to_pic), cv2.IMREAD_GRAYSCALE)
imgar4 = cv2.resize(imgax4, (size, size))

ttX = []
for feature in imgar4:
    ttX.append(feature)
          
ttX = np.array(ttX).reshape(-1, size, size, 1)
model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(64, kernel_size = [3,3], padding = 'same', activation = 'relu', input_shape = ttX.shape[1:]),
            tf.keras.layers.Conv2D(64, kernel_size = [3,3], padding = 'same', activation = 'relu'),
            tf.keras.layers.MaxPool2D(pool_size = [3,3]),
            tf.keras.layers.Conv2D(128, kernel_size = [3,3], padding = 'same', activation = 'relu'),
            tf.keras.layers.MaxPooling2D(3,3),
            tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2,2), 
            tf.keras.layers.Conv2D(32, (3,3), activation='relu'), 
            tf.keras.layers.MaxPooling2D(2,2),
            tf.keras.layers.Dense(1024, activation = 'relu'),
            tf.keras.layers.Dense(512, activation = 'relu'),
            tf.keras.layers.Flatten(), 
            tf.keras.layers.Dense(1013, activation='relu'), 
            tf.keras.layers.Dense(36, activation='softmax')  
    ])
      
model.summary()
model.compile(optimizer=RMSprop(lr=0.001),
            loss='sparse_categorical_crossentropy',
            metrics = ['acc'])
model.load_weights("Pretrained_Model/cp.ckpt")
#loss,acc = model.evaluate(tX, tY)
#print("Restored model, accuracy: {:5.2f}%".format(100*acc))

zzx = ""
while zzx != "exit":
      path_to_pic = input("Enter the path to the image you would like to predict: ")
      imgax4 = cv2.imread(os.path.join(path_to_pic), cv2.IMREAD_GRAYSCALE)
      imgar4 = cv2.resize(imgax4, (size, size))

      ttX = []
      for feature in imgar4:
        ttX.append(feature)
          
      ttX = np.array(ttX).reshape(-1, size, size, 1)      
      eer = model.predict(ttX)
      z = np.where(eer == 1)
      if z[1]>9:
          prediction = chr((z[1]-10) + 65)
          print(prediction)
      else:
          prediction = int(z[1])
          print(prediction)
      zzx = input("Enter exit to stop or return to keep predicting.")

