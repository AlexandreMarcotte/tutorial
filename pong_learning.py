import tensorflow as tf
import cv2
import pong
import numpy as np
import random
from collections import deque

#defining hyperparameters
ACTIONS = 3
#learning rate
GAMMA = 0.99
#update our gradient or training time
INITIAL_EPSILON = 1.0
FINAL_EPSILON = 0.005
#how many frames to anneal epsilon
EXPLORE = 500_000
OBSERVE = 50_000
REPLAY_MEMORY = 50_000
#batch size
BATCH = 100

#create TF graph
def createGraph():
    #first convolutional layer, bias vector
    W_conv1 = tf.Variable(tf.zeros([8, 8, 4, 32]))
    b_conv1 = tf.Variable(tf.zeros[32])

    #second
    W_conv2 = tf.Variable(tf.zeros[4,4,32,64])
    b_conv2 = tf.Variable(tf.zeros[64])

    #third
    W_conv3 = tf.Variable(tf.zeros[3,3,64, 64])
    b_conv3 = tf.Variable(tf.zeros[64])

    #fourth
    W_fc4 = tf.Variable(tf.zeros[784, ACTIONS])
    b_fc4 = tf.Variable(tf.zeros[784])

    #LAST LAYER
    W_fc5 = tf.Variable(tf.zeros[784, ACTIONS])
    b_fc5 = tf.Variable(tf.zeros[[Actions]])

    #input for pixel data
    s = tf.placeholder('float' [None, 84, 84 , 84])

    #compute RELU actiovation function
    #on 2d convolutions
    #given 4D inputs and filter tensors

    conv1 = tf.nn.relu(tf.nn.conv2d(s, W_conv1, strides[1, 4, 4, 1], padding = 'VALID') + b_conv1)
    conv2 = tf.nn.relu(tf.nn.conv2d(s, W_conv2, strides[1, 4, 4, 1], padding = 'VALID') + b_conv1)
    conv3 = tf.nn.relu(tf.nn.conv2d(s, W_conv3, strides[1, 4, 4, 1], padding = 'VALID') + b_conv1)

    conv3_flat = tf.reshape(conv3, [-1, 3136])
    fc4 = tf.nn.relu(tf.matmul(conv3_flat, W_fc4 + b_fc4))
    fc5 = tf.matmul(fc5, W_fc5) + b_fc5

    return s, fc5

def main():

    #create session
    sess = tf.InteractiveSession()
    #input player and ouroutput layer
    inp, out = CreateGraph()
    trainGraph(inp, out, sess)

if __name__ == '__main__':
    main()