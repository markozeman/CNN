# ID=7e6

import requests
import os
import h5py
import cv2
import numpy as np
from os import listdir
from os.path import isfile, join
from PIL import Image
from urllib.request import urlretrieve
from lxml import html
from keras.models import Sequential, load_model
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from keras.utils import to_categorical


def show_image(name):
    """
    Shows image.
    :param name: image name
    :return: None
    """
    img = Image.open(name)
    img.show()


def download_images():
    """
    Downloads images from server and edits them to folders.
    :return: None
    """
    base_url = 'http://file.biolab.si/files/tmp/yeast_images/'
    directories = ['test/', 'train/0-CYTOPLASM/', 'train/1-NUCLEUS/', 'train/2-ER/', 'train/3-MITOCHONDRIA/']
    for d in directories:
        if not os.path.exists(d):
            os.makedirs(d)
        page = requests.get(base_url + d)
        tree = html.fromstring(page.content)
        links = tree.xpath('//a/text()')[1:]
        for l in links:
            urlretrieve(base_url + d + l, './' + d + l)


def downsize_image(name, size):
    """
    Downsizes image to specified size.
    :param name: image name
    :param size: new image size
    :return: None
    """
    img = Image.open(name)
    img = img.resize(size, Image.ANTIALIAS)
    img.save(name)


def reduce_images(size):
    """
    Reduces all images to specified size.
    :param size: new image size
    :return: None
    """
    directories = ['test/', 'train/0-CYTOPLASM/', 'train/1-NUCLEUS/', 'train/2-ER/', 'train/3-MITOCHONDRIA/']
    for d in directories:
        images = [f for f in listdir(d) if isfile(join(d, f))]
        for im in images:
            downsize_image(d + im, size)


def shuffle_copies(a, b):
    """
    Shuffles data (same permutation for a and b) and returns their copies.
    :param a: array or matrix
    :param b: array or matrix
    :return: shuffled copies of a and b
    """
    p = np.random.permutation(len(a))
    return a[p], b[p]


def standardize(data):
    """
    Standardizes data (images).
    :param data: array of images
    :return: standardized images
    """
    for i in range(len(data)):
        data[i] = (data[i] - data[i].mean()) / data[i].std()
    return data


def read_images():
    """
    Reads images from directories and saves it in arrays.
    :return: X_train, y_train, X_test
    """
    directories = ['train/0-CYTOPLASM/', 'train/1-NUCLEUS/', 'train/2-ER/', 'train/3-MITOCHONDRIA/']
    images_array = []
    targets = np.array([])
    for d in directories:
        images = [f for f in listdir(d) if isfile(join(d, f))]
        target = int(d.split('/')[1][0])
        targets = np.concatenate((targets, np.full(len(images), target)))
        for im in images:
            i = cv2.imread(d + im)     # cv2.imread() returns a numpy array in BGR not RGB
            images_array.append(i)
    X_train = np.array(images_array)

    X_train, targets = shuffle_copies(X_train, targets)
    y_train = to_categorical(targets)

    d = 'test/'
    X_test = []
    images = [f for f in listdir(d) if isfile(join(d, f))]
    images.sort()
    for im in images:
        i = cv2.imread(d + im)
        X_test.append(i)
    X_test = np.array(X_test)

    X_train = standardize(X_train)
    X_test = standardize(X_test)

    return X_train, y_train, X_test


def write2file(filename, vector):
    """
    Writes vector to file.
    :param filename: name of the file
    :param vector: vector of numbers to write to a file
    :return: None
    """
    s = '\n'.join(list(map(lambda x: str(x), vector)))
    with open(filename, 'w') as f:
        f.write(s)


class ConvolutionalNeuralNetwork:
    """
    Implementation of convolutional neural network in Keras for classification of yeast images.
    """
    def __init__(self, download=False, size=(128, 128)):
        """
        Downloads images from server and reduces their size.
        Saves CNN model.
        :param size: new size of images
        """
        if download:
            download_images()
            reduce_images(size)
        self.model = self.cnn_model()

    @staticmethod
    def cnn_model():
        """
        Created CNN model in Keras.
        :return: model
        """
        input_shape = (128, 128, 3)
        activation = 'relu'
        padding = 'same'
        n_classes = 4

        model = Sequential()
        model.add(Conv2D(64, (3, 3), padding=padding, activation=activation, input_shape=input_shape))
        model.add(Conv2D(64, (3, 3), padding=padding, activation=activation))
        model.add(MaxPooling2D(pool_size=(5, 5)))

        model.add(Conv2D(128, (3, 3), padding=padding, activation=activation))
        model.add(Conv2D(128, (3, 3), padding=padding, activation=activation))
        model.add(MaxPooling2D(pool_size=(5, 5)))

        model.add(Conv2D(256, (3, 3), padding=padding, activation=activation))
        model.add(Conv2D(256, (3, 3), padding=padding, activation=activation))
        model.add(Conv2D(256, (3, 3), padding=padding, activation=activation))
        model.add(MaxPooling2D(pool_size=(3, 3)))

        model.add(Flatten())
        model.add(Dense(512))
        model.add(Dropout(0.5))
        model.add(Dense(n_classes, activation='softmax'))

        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        model.summary()
        return model

    def fit_predict(self, epochs=50):
        """
        Fits the model on training data and predicts classes for test data.
        Writes results to output text file.
        :param epochs: number of training epochs
        :return: self
        """
        X_train, y_train, X_test = read_images()

        checkpoint = ModelCheckpoint('model.hdf5', monitor='val_acc', verbose=0, save_best_only=True)
        self.model.fit(X_train, y_train, epochs=epochs, validation_split=0.25, verbose=2, callbacks=[checkpoint])

        self.model = load_model('model.hdf5')

        classes = self.model.predict(X_test)
        predictions = np.argmax(classes, axis=1)

        write2file('output.txt', predictions)
        return self


if __name__ == '__main__':
    cnn = ConvolutionalNeuralNetwork()
    cnn.fit_predict()
