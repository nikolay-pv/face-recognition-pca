import cv2
import fnmatch
import os
import numpy as np

from fileManipulations import get_label
from vectorOperations import normalize


width = 168
height = 216

def read_image(from_file):
    filename = os.path.abspath(from_file)
    img_label = get_label(from_filename=filename)
    nbDim = height*width
    X = np.zeros((1, nbDim))
    bimg = cv2.imread(filename, 0)
    img = cv2.resize(bimg, (width, height))
    img = cv2.equalizeHist(img)
    img = normalize(img)
    # cv2.imshow('image', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    X[0, :] = img.flatten()
    return X, img_label


def read_all_images(from_directory):
    directory = os.path.abspath(from_directory)
    filenames = fnmatch.filter(os.listdir(directory), '*.png')
    nbDim = height*width
    nb_images = len(filenames)
    X = np.zeros((nb_images, nbDim))
    labels = np.zeros(nb_images, dtype=int)
    for i, filename in enumerate(filenames):
        filename = os.path.join(directory, filename)
        X[i, :], labels[i] = read_image(filename)
    return X, labels, filenames

