import numpy as np


def distance(X, Y):
    return np.sqrt(np.sum(np.power((X-Y), 2)))


def project(W, X, mu):
    """
        Shifts X by mean value mu and project it to vector space W.
    """
    return np.dot(X - mu, W)


def reconstruct(W, Y, mu):
    """
        Reconstruct an image based on its PCA-coefficients Y, the eigenvectors W and the average mu.
    """
    return np.dot(Y, W.T) + mu


def normalize(img):
    """
        Normalize an image such that colour values min=0 and max=255.
        Converts type of array's values to np.uint8
    """
    minimum, maximum = np.min(img), np.max(img)
    img = img - float(minimum)
    img = img/(maximum - minimum)
    img = img*255.
    return np.asarray(img, dtype=np.uint8)
