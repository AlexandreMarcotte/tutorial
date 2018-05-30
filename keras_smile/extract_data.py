import matplotlib.pyplot as plt
from PIL import Image
import os
from os import listdir
from os.path import isfile, join
import numpy as np

def extract_img(
        path: str,
        class_num: int,
        subsample: int=1,
        split_thresh: float=0.8,
        SHOW_SUBSET: bool=False) -> list:
    """
    extract_img(path=r'C:/Users/alexa/Documents/CODING/2018/tutorial/keras_smile',
                class_num=1, subsample=1, split_thresh=0.8, SHOW_SUBSET=False)

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
    :subsample: (int)
        Subsampling the image to classify
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
        # im = im.convert('1', dither=Image.NONE)

        # append the np.array img to a list
        if i <= split_pos:
            print(np.array(im).shape)
            x_train.append(np.array(im)[::subsample, ::subsample])
        else:
            x_test.append(np.array(im)[::subsample, ::subsample])

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