#training model for self driving car

# Import Libraries #

import os
import cv2
import pandas as pd
import sys
import random
import numpy as np
from imgaug import augmenters as iaa
import keras
from keras.models import Sequential
from keras.layers import Convolution2D, Dropout, Flatten, Dense
from keras.optimizers import Adam


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
        #print(type(self.data_set.head()))

        #tuple of parsed set, and labels
        self.test = self.parse_images(self.test)
        self.train = self.parse_images(self.train)
        #print(self.image_to_dataframe(self.img_preprocess(cv2.imread(self.test[0][0])), self.test[0][1]))
        #print(self.test[:5])
    """
    def parse_images(self, dataset):
        processed_dataset = []
        labels = []
        count = 0
        for point in dataset:
            if count == 5:
                break
            count += 1
            img = cv2.imread(point[0])
            if img is None:
            #    print(img, point)
                continue
            image_dataframe = self.image_to_dataframe(self.img_preprocess(img), point[1])
            processed_dataset.append(image_dataframe[0])
            labels.append(image_dataframe[1])
        return (processed_dataset, labels)
    """
    def parse_images(self, dataset):
        processed_dataset = []
        labels = []
        for point in dataset:
            img = cv2.imread(point[0])
            #image_dataframe = self.image_to_dataframe(self.img_preprocess(img), point[1])
            processed_dataset.append(self.img_preprocess(img))
            #processed_dataset.append(img)
            labels.append(point[1])
        return (processed_dataset, labels)

    #not in use right now
    def image_to_dataframe(self, img, label):
        processed_img = []
        for y in range(len(img)):
            for x in range(len(img[y])):
                rgb = 65536 * img[y][x][0] + 256 * img[y][x][1] + img[y][x][2]
                processed_img.append(rgb)

        return (label, processed_img)

    def parse(self, data, testdir):
        def split_path(path):
            return path.split('\\')[-1]

        data['center'] = data['center'].apply(split_path)
        data['left'] = data['left'].apply(split_path)
        data['right'] = data['right'].apply(split_path)

        ###### test
        data.drop(['left'], axis=1)
        data.drop(['right'], axis=1)
        ######


        paths, direction = self.parse_image_path(data, testdir+'/IMG/')
        self.test, self.train = self.split_data(paths, direction)

    def parse_image_path(self, data, testdir):
        paths = []
        directions = []
        for i in range(len(data)):
            data_point = data.iloc[i]
            center = data_point[0]
            #left = data_point[1]
            #right = data_point[2]

            steer = float(data_point[3])
            paths.append(testdir+center.strip())
            directions.append(steer)

            #paths.append(testdir+left.strip())
            #directions.append(steer+0.15)

            #paths.append(testdir+left.strip())
            #directions.append(steer-0.15)
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

    def img_preprocess(self, img):
        img = img[60:135,:,:]
        img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        img = cv2.GaussianBlur(img,  (3, 3), 0)
        img = cv2.resize(img, (200, 66))
        #img = img/255
        return img

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

    def train_model(self):
        model = self.nvidia_model_builder()
        optimizer = Adam(lr=1e-3)
        model.compile(loss='mse', optimizer=optimizer)
        model.fit(np.asarray(self.train[0]), np.asarray(self.train[1]), epochs=20)
        predictions = model.predict(np.asarray(self.test[0]))
        print(len(predictions))
        acc = 0
        for i in range(len(predictions)):
            print(predictions[i][0], self.test[1][i])
            val = predictions[i].astype(np.float)
            lab = self.test[1][i].astype(np.float)
            if abs(val - lab) < .05:
                acc += 1
        print(acc/len(predictions))



    def nvidia_model_builder(self):
        model = Sequential()
        model.add(Convolution2D(3, (5, 5), strides=(2,2), input_shape=(66, 200, 3), activation='elu'))
        #model.add(Convolution2D(3, (5, 5), strides=(2,2), input_shape=(160, 320, 3), activation='elu'))
        model.add(Convolution2D(24, (5, 5), strides=(2,2), input_shape=(31, 98, 24), activation='elu'))
        model.add(Convolution2D(36, (5, 5), strides=(2,2), input_shape=(14, 47, 36), activation='elu'))
        model.add(Convolution2D(48, (3, 3), input_shape=(5, 22, 48), activation='elu'))
        model.add(Convolution2D(64, (3, 3), input_shape=(3, 20, 64), activation='elu'))
        model.add(Flatten())
        model.add(Dense(1164, activation='elu'))
        model.add(Dense(100, activation='elu'))
        model.add(Dense(50, activation='elu'))
        model.add(Dense(10, activation='elu'))
        model.add(Dense(1))

        return model

if __name__ == "__main__":
    seed = 12345
    training_perc = 0.70
    testset = "Drive-1/driving_log.csv"
    dir = "Drive-1"
    model = SD_NN(testset, dir, seed, training_perc)
    model.train_model()
