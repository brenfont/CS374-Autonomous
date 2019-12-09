#training model for self driving car

import os
import csv
import pandas as pd
import sys
import random
import numpy as np
from imgaug import augmenters as iaa


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

if __name__ == "__main__":
    seed = 12345
    training_perc = 0.75
    testset = "Drive-1/driving_log.csv"
    dir = "Drive-1"
    model = SD_NN(testset, dir, seed, training_perc)
