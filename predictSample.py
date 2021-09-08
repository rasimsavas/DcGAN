# -*- coding: utf-8 -*-
"""
Created on Wed Aug 18 00:07:27 2021

@author: savas
"""



import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
from zipfile import ZipFile
from tensorflow.python.client import device_lib 
import matplotlib.image as mpimg 
from matplotlib.pyplot import imshow
import os
from tensorflow.python.client import device_lib


os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
dir = r"C:\Users\savas\Desktop\DCGAN\savedModel"
x = tf.saved_model.load(dir)


img = x.generator(tf.random.normal(shape=(3,512)))
for i in range(3):

   
    imgary = keras.preprocessing.image.array_to_img(img[i])

    imgary.save(r"C:\Users\savas\Desktop\DCGAN\imgmodel\generated_img_%d.png" % (i))
