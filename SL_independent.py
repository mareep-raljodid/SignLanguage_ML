
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

# Download the ASL dateset from here: https://drive.google.com/open?id=1apmXyY8OQx68b4-2G9Mttgrml7bldWfq
# Don't forget to update the path on the below Dir variable

Dir = "asl"
catg = ["0","1","2","3","4","5","6","7","8","9","10","11","12","13",
"14","15","16","17","18","19","20","21","22","23","24","25","26","27",
"28","29","30","31","32","33","34","35"]

train = []
size = 200
def create_tr():
    for c in catg:
        path=os.path.join(Dir, c)
        class_nnn= catg.index(c)
        for img in os.listdir(path):
            imgax = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE) 
            imgar = cv2.resize(imgax, (size, size))
            train.append([imgar, class_nnn])


create_tr()

print("Number of elements in training set: ", len(train))
random.shuffle(train)

tX = []
tY = []

for features, label in train:
    tX.append(features)
    tY.append(label)
    
tX = np.array(tX).reshape(-1, size, size, 1) #1 not 3
tX = tX / 255.0

class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('acc')>0.99):
      print("\nReached 99% accuracy so cancelling training!")
      self.model.stop_training = True

# Below are the callBacls for saving the model and 99% accuracy Epocs Callback


callbacks1 = myCallback()
checkpoint_path = "training_data/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)


model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(64, kernel_size = [3,3], padding = 'same', activation = 'relu', input_shape = tX.shape[1:]),
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

model.fit(tX, tY, batch_size=25, validation_split=0.10, epochs = 10, callbacks=[callbacks1, cp_callback])

# This code comented out below is for restoration process of the Model trained by me. 
# You're welcome :) 

'''
model.load_weights("Pretrained_Model/cp.ckpt")
loss,acc = model.evaluate(tX, tY)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))
'''

# Now comes the fun part!! 
# Predict the hand sign by editing the path_to_pic valiable below, 
# don't forget that darker the backgroung, more accurate prediction!!

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
               

