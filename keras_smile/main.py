from tutorial.keras_smile import collecting_data
from tutorial.keras_smile import extract_data
from tutorial.keras_smile import classifying_data
import os
import numpy as np
import matplotlib.pyplot as plt

def main():
    "COLLECTING the data from the webcam"
    collect_path = r'C:\Users\alexa\Documents\CODING\2018\tutorial\keras_smile/'
    if False:
        collecting_data.collect_data(collect_path)

    "EXTRACT Images (from webcam) to be classified"
    # # Path of the images folder to classify
    base_path = r'C:\Users\alexa\Documents\CODING\2018\tutorial\keras_smile'
    r_im_path = os.path.join(base_path, 'right_hand')
    l_im_path = os.path.join(base_path, 'left_hand')
    x_train = []
    y_train = []
    x_test = []
    y_test = []

    for class_num, im_type in enumerate([r_im_path, l_im_path]):
        (x_tr, y_tr), (x_te, y_te) = extract_data.extract_img(im_type,
                                                              class_num,
                                                              subsample=4,
                                                              split_thresh=0.8,
                                                              SHOW_SUBSET=True)
        x_train.extend(x_tr)
        y_train.extend(y_tr)
        x_test.extend(x_te)
        y_test.extend(y_te)

    "CREATE simple images"
    # x_train = []
    # y_train = []
    # x_test = []
    # y_test = []
    # for i in range(100):
    #     x_train.append(np.ones((50, 50)))
    #     y_train.append(0)
    #     # plt.imshow(x_train[i])
    #     # plt.savefig(os.path.join(r_im_path, f'r{i}'))
    # for i in range(20):
    #     x_test.append(np.zeros((50,50)))
    #     y_test.append(1)
    #     # plt.imshow(x_test[i])
    #     # plt.savefig(os.path.join(l_im_path, f'l{i}'))


    # Convert the list to numpy array is required for the training
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_test = np.array(x_test)
    y_test = np.array(y_test)

    "LEARNING phase"
    classifying_data.train_classifier(x_train, y_train, x_test, y_test)


main()