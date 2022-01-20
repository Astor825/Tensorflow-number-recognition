import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import keras
from tensorflow.keras import layers
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Activation, BatchNormalization, AveragePooling2D
from tensorflow.keras.optimizers import SGD, RMSprop, Adam
import tensorflow_datasets as tfds  
import tensorflow as tf
import logging
import numpy as np
import time


def mnist_make_model(image_w: int, image_h: int):
 
   model = Sequential()
   model.add(Dense(784, activation='relu', input_shape=(image_w*image_h,)))
   model.add(Dense(10, activation='softmax'))
   model.compile(loss='categorical_crossentropy', optimizer=RMSprop(), metrics=['accuracy'])
   return model

def mnist_mlp_train(model):
   (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
   image_size = x_train.shape[1]
   train_data = x_train.reshape(x_train.shape[0], image_size*image_size)
   test_data = x_test.reshape(x_test.shape[0], image_size*image_size)
   train_data = train_data.astype('float32')
   test_data = test_data.astype('float32')
   train_data /= 255.0
   test_data /= 255.0
   num_classes = 10
   train_labels_cat = keras.utils.to_categorical(y_train, num_classes)
   test_labels_cat = keras.utils.to_categorical(y_test, num_classes)
   print("Training the network...")
   t_start = time.time()

   model.fit(train_data, train_labels_cat, epochs=8, batch_size=64, verbose=1, validation_data=(test_data, test_labels_cat))


def mlp_digits_predict(model, image_file):
   image_size = 28
   img = keras.preprocessing.image.load_img(image_file, target_size=(image_size, image_size), color_mode='grayscale')
   img_arr = np.expand_dims(img, axis=0)
   img_arr = 1 - img_arr/255.0
   img_arr = img_arr.reshape((1, image_size*image_size))
   result = model.predict([img_arr])
   return result[0]

model = mnist_make_model(image_w=28, image_h=28)
mnist_mlp_train(model)
model.save('mlp_digits_28x28.h5')

model = tf.keras.models.load_model('mlp_digits_28x28.h5')
print(mlp_digits_predict(model, 'digit_1.png'))
print("\n")
print(mlp_digits_predict(model, 'digit_2.png'))
print("\n")
print(mlp_digits_predict(model, 'digit_3.png'))
print("\n")
print(mlp_digits_predict(model, 'digit_4.png'))
print("\n")
print(mlp_digits_predict(model, 'digit_5.png'))
print("\n")
print(mlp_digits_predict(model, 'digit_6.png'))
print("\n")
print(mlp_digits_predict(model, 'digit_7.png'))
print("\n")
#print(mlp_digits_predict(model, 'digit_8.png'))
#print("\n")
#print(mlp_digits_predict(model, 'digit_9.png'))
#print("\n")
#print(mlp_digits_predict(model, 'digit_10.png'))
