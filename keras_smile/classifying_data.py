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



def extract_img(
        path:str,
        class_num:int,
        split_thresh:float = 0.8,
        SHOW_SUBSET:bool = False) -> list:
    """
    extract_img(path=r'C:\Users\alexa\Documents\CODING\2018\tutorial\keras_smile',
                class_num=1, split_thresh=0.8, SHOW_SUBSET=False)

    This function extract the images in a folder and split them into
    x_train or x_test depending on the split threshold proportion.
    It then create the class value for the corresponding images in y_train
    and y_test

    Parameters:
    ----------
    :path: (str)
        The base path where the images to classify are located
    :class_num: (int)
        The class value of the current image type
    :split_thresh: (float)
        The proportion of image to put inside the training variable
    :SHOW_SUBSET: (bool)
        If you want to show a small subset of the images in the folder

    Returns:
    -------
    :x_train: (list of list)
        List of the images of one class for the training
    :y_train: (list)
        List of class value of the corresponding image in x_train
    :x_test: (list of list)
        List of the images of one class for the test
    :y_test: (list)
        List of class value of the corresponding image in x_test

    """
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    i = 0
    # list all the img in the file
    im_name_list = listdir(path)
    split_pos = int(split_thresh * len(im_name_list))
    for i, im_name in enumerate(im_name_list):
        # create the url for every image
        url = os.path.join(path, im_name)
        im = Image.open(url)
        # convert img to gray scale
        im = im.convert('1', dither=Image.NONE)
        # append the np.array img to a list
        if i <= split_pos:
            x_train.append(np.array(im))
        else:
            x_test.append(np.array(im))

        if SHOW_SUBSET:
            if i < 5:
                plt.imshow(im)
                plt.show()
            i += 1

        # Add a classification number for every images in the
        # train and test set
        y_train = [class_num for _ in range(len(x_train))]
        y_test = [class_num for _ in range(len(x_test))]
    return (x_train, y_train), (x_test, y_test)


# Path of the images folder to classify
base_path = r'C:\Users\alexa\Documents\CODING\2018\tutorial\keras_smile'
r_im_path = os.path.join(base_path, 'right_hand')
l_im_path = os.path.join(base_path, 'left_hand')
x_train = []
y_train = []
x_test = []
y_test = []

for class_num, im_type in enumerate([r_im_path, l_im_path]):
    (x_tr, y_tr), (x_te, y_te) = extract_img(im_type, class_num,
                                             split_thresh=0.8)

    x_train.extend(x_tr)
    y_train.extend(y_tr)
    x_test.extend(x_te)
    y_test.extend(y_te)





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

# # # the data, split between train and test sets
# (x_train, y_train), (x_test, y_test) = split_list([all_r_im, all_l_im], 0.8)
#
# batch_size = 12
# num_classes = 2
# epochs = 1
# # input image dimensions
# img_rows, img_cols = 480, 640
#
# if K.image_data_format() == 'channels_first':
#     x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
#     x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
#     input_shape = (1, img_rows, img_cols)
# else:
#     x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
#     x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
#     input_shape = (img_rows, img_cols, 1)
#
# x_train = x_train.astype('float32')
# x_test = x_test.astype('float32')
# x_train /= 255
# x_test /= 255
# print('x_train shape:', x_train.shape)
# print(x_train.shape[0], 'train samples')
# print(x_test.shape[0], 'test samples')
#
# # convert class vectors to binary class matrices
# y_train = keras.utils.to_categorical(y_train, num_classes)
# y_test = keras.utils.to_categorical(y_test, num_classes)
#
# model = Sequential()
# model.add(Conv2D(32, kernel_size=(3, 3),
#                  activation='relu',
#                  input_shape=input_shape))
# model.add(Conv2D(64, (3, 3), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))
# model.add(Flatten())
# model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(num_classes, activation='softmax'))
#
# model.compile(loss=keras.losses.categorical_crossentropy,
#               optimizer=keras.optimizers.Adadelta(),
#               metrics=['accuracy'])
#
# model.fit(x_train, y_train,
#           batch_size=batch_size,
#           epochs=epochs,
#           verbose=1,
#           validation_data=(x_test, y_test))
# score = model.evaluate(x_test, y_test, verbose=0)
#
#
# print('Test loss:', score[0])
# print('Test accuracy:', score[1])
