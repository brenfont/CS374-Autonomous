#training model for self driving car

# Import Libraries #

import os
import csv
import pandas as pd
import sys
import random
import numpy as np
from imgaug import augmenters as iaa
import keras
from keras.models import Sequential
from keras.layers import Convolution2D, Dropout, Flatten, Dense


class SD_NN:
    def __init__(self, testset, testdir, seed, training_perc):
        columns = ['center', 'left', 'right', 'steering', 'throttle', 'reverse', 'speed']
        self.data_set = pd.read_csv(testset, names = columns)
        #pd.set_option('display.max_colwidth', -1)
        self.training_perc = training_perc
        np.random.seed(seed)

        self.test = []
        self.train = []

        self.parse(self.data_set, testdir)
        #print(self.data_set.head()
        print(self.test[:5])


    def parse(self, data, testdir):
        def split_path(path):
            return path.split('\\')[-1]
        data['center'] = data['center'].apply(split_path)
        data['left'] = data['left'].apply(split_path)
        data['right'] = data['right'].apply(split_path)
        paths, direction = self.parse_image_path(data, testdir+'/IMG/')
        self.test, self.train = self.split_data(paths, direction)

    def parse_image_path(self, data, testdir):
        paths = []
        directions = []
        for i in range(len(data)):
            data_point = data.iloc[i]
            center = data_point[0]
            left = data_point[1]
            right = data_point[2]

            steer = float(data_point[3])
            paths.append(testdir+center.strip())
            directions.append(steer)

            paths.append(testdir+left.strip())
            directions.append(steer+0.15)

            paths.append(testdir+left.strip())
            directions.append(steer-0.15)
        paths = np.asarray(paths)
        directions = np.asarray(directions)
        return paths, directions

    def split_data(sel, paths, directions):
        shuffle_set = []
        for i in range(len(paths)):
            tup = (paths[i], directions[i])
            shuffle_set.append(tup)
        shuffle_set = np.array(shuffle_set)

        np.random.shuffle(shuffle_set)

        splitind = int(len(shuffle_set) * training_perc)

        train = shuffle_set[:splitind]
        test = shuffle_set[splitind:]
        return train, test

    def zoom(im):
        zoom = iaa.Affine(scale=(1, 1.3))
        im = zoom.augment_image(im)
        return im

    #def train(self):


    '''
    Nvidia Self Driving Car Nueral Network - We are building the implementation of a self driving neural net described in the Nvidia Developer blog

    "We train the weights of our network to minimize the mean-squared error between the steering command output by the network, and either the command of the
    human driver or the adjusted steering command for off-center and rotated images (see “Augmentation”, later). Figure 5 shows the network architecture, which
    consists of 9 layers, including a normalization layer, 5 convolutional layers, and 3 fully connected layers. The input image is split into YUV planes and
    passed to the network." - Nvidia Developer Blog

    https://devblogs.nvidia.com/deep-learning-self-driving-cars/
    '''

    # Nvidia Network Architecture Implemented Using Keras #

    def nvidia_model_builder():

        model = Sequential()
        model.add(Convolution2D(3, 5, 5, strides=(2,2), input_shape = (66, 200, 3), activation = 'elu'))
        model.add(Convolution2D(24, 5, 5, strides=(2,2), input_shape = (31, 98, 24), activation = 'elu'))
        model.add(Convolution2D(36, 5, 5, strides=(2,2), input_shape = (14, 47, 36), activation = 'elu'))
        model.add(Convolution2D(48, 3, 3, input_shape = (5, 22, 48), activation = 'elu'))
        model.add(Convolution2D(64, 3, 3, input_shape = (3, 20, 64), activation = 'elu'))
        model.add(Flatten())
        model.add(Dense(1164), activation = 'elu')
        model.add(Dense(100), activation = 'elu')
        model.add(Dense(50), activation = 'elu')
        model.add(Dense(10), activation = 'elu')
        model.add(Dense(1))

        return model

if __name__ == "__main__":
    seed = 12345
    training_perc = 0.75
    testset = "Drive-1/driving_log.csv"
    dir = "Drive-1"
    model = SD_NN(testset, dir, seed, training_perc)
