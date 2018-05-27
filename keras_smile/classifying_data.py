#CodingANewWorld
from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

import matplotlib.pyplot as plt
from PIL import Image
import os
from os import listdir
from os.path import isfile, join
import numpy as np

# Path of the images folder to classify
base_path = r'C:\Users\alexa\Documents\CODING\2018\tutorial\keras_smile'
r_im_path = os.path.join(base_path, 'right_hand')
l_im_path = os.path.join(base_path, 'left_hand')

def extract_img(path: str) -> list:
    """
    Parameters:
    ----------
    :path:
        the base path where the images to classify are located
    Returns:
    -------

    """
    all_img = []
    i = 0
    # list all the img in the file
    im_name_list = listdir(path)
    for im_name in im_name_list:
        # create the url
        url = os.path.join(path, im_name)
        im = Image.open(url)
        # convert img to gray scale
        im = im.convert('1', dither=Image.NONE)
        # append the np.array img to a list
        all_img.append(np.array(im))
        # if i < 10:
        #     plt.imshow(im)
        #     plt.show()
        # i += 1
    return all_img

all_r_im = extract_img(r_im_path)
all_l_im = extract_img(l_im_path)


def split_list(class_list, thresh): # TODO ALEXM: Split the image directly at the extraction! 
    """
    Split the differents class data into train and test set and stick them
    one after the other

    Parameters:
    ----------
        :data_list: list of list
            a list of all the list contaning the different images type that
            need to be classify
        :thresh: float
            a threshold value that indicate where to split the lists
    Returns:
    -------

    """
    x_train = []
    y_train = []
    x_test = []
    y_test = []

    for i, one_class in enumerate(class_list):
        split_pos = int(thresh * len(one_class))
        # Training split
        x_tr = one_class[:split_pos]
        x_train += x_tr
        y_train += list(np.ones(len(x_tr)) * i)
        # Testing split
        x_te = one_class[split_pos:]
        x_test += x_te
        y_test += list(np.ones(len(x_te)) * i)
        # convert to numpy array
        x_train, y_train, x_test, y_test = (np.array(x_train), np.array(y_train),
                                           np.array(x_test), np.array(y_test))
    return (x_train, y_train), (x_test, y_test)

# # the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = split_list([all_r_im, all_l_im], 0.8)

batch_size = 12
num_classes = 2
epochs = 1
# input image dimensions
img_rows, img_cols = 480, 640

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)


print('Test loss:', score[0])
print('Test accuracy:', score[1])
