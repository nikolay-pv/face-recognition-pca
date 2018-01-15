import cv2
import numpy as np
from vectorOperations import distance, normalize, project, reconstruct


class PCAFaceRecognizer(object):
    def __init__(self, X=None, labels=None, nb_components=0):
        self.nb_components = nb_components
        self.eigenvectors = []
        self.labels = []
        self.mu = []
        self.projections = []
        if (X is not None) and (labels is not None):
            self.compute(X, labels)

    def predict(self, Y):
        min_dist = np.finfo('float').max
        min_class = -1
        projection = project(self.eigenvectors, Y.reshape(1, -1), self.mu)
        distances = np.zeros(len(self.projections))
        for i in range(len(self.projections)):
            distances[i] = distance(self.projections[i], projection)
            if distances[i] < min_dist:
                min_dist = distances[i]
                min_class = self.labels[i]
        return min_class, min_dist, distances

    def print_eigenface(self, face):
        projection = project(self.eigenvectors, face, self.mu)
        face_new = reconstruct(self.eigenvectors, projection, self.mu)
        # img = cv2.equalizeHist(face_new)
        img = normalize(face_new)
        img = img.reshape(150, 150)
        window_name = "Projected and reconstructed img"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.imshow(window_name, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def compute(self, X, labels):
        eigenvalues, self.eigenvectors, self.labels, self.mu = self.pca(X, labels, self.nb_components)
        for xi in X:
            self.projections.append(project(self.eigenvectors, xi.reshape(1, -1), self.mu))
        return

    def pca(self, X, labels, nb_components=0):
        """
        PCA analysis on set of samples X
        :param X:                np.array containing the samples of shape:
                                 number samples, number dimensions of each sample
        :param labels:           labels of the samples
        :param nb_components:    number of components to return
        :return:                 return the nb_components largest eigenvalues and eigenvectors
                                 of the covariance matrix and return the mean value
        """
        n, _ = X.shape
        if (nb_components <= 0) or (nb_components > n):
            nb_components = n
        # Substract the mean
        mu_X = X.mean(axis=0)
        for i in range(n):
            X[i, :] = X[i, :] - mu_X
        # Compute covariant matrix
        cov_matrix = np.dot(X, X.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        eigenvectors = np.dot(X.T, eigenvectors)
        for i in range(n):
            eigenvectors[:, i] = eigenvectors[:, i]/np.linalg.norm(eigenvectors[:, i])
        index = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[index]
        eigenvectors = eigenvectors[:, index]
        # Select the given number of components
        eigenvalues = eigenvalues[0:nb_components].copy()
        eigenvectors = eigenvectors[:, 0:nb_components].copy()
        return [eigenvalues, eigenvectors, labels, mu_X]

