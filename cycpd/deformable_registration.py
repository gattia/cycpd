from builtins import super
import numpy as np
import time
from .expectation_maximization_registration import expectation_maximization_registration

def gaussian_kernel(Y, beta):
    diff = Y[None,:,:] - Y[:,None,:]
    diff = diff**2
    diff = np.sum(diff, 2)
    return np.exp(-diff / (2 * beta**2))

class deformable_registration(expectation_maximization_registration):
    def __init__(self, alpha=2, beta=2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha         = 2 if alpha is None else alpha
        self.beta          = 2 if beta is None else beta
        self.W             = np.zeros((self.M, self.D))
        self.G             = gaussian_kernel(self.Y, self.beta)

    def update_transform(self):
        tic = time.time()
        A = np.dot(np.diag(self.P1), self.G) + self.alpha * self.sigma2 * np.eye(self.M)
        toc = time.time()
        self.A_times.append(toc-tic)
        tic = time.time()
        B = np.dot(self.P, self.X) - np.dot(np.diag(self.P1), self.Y)
        toc = time.time()
        self.B_times.append(toc - tic)
        tic = time.time()
        self.W = np.linalg.solve(A, B)
        toc = time.time()
        self.solve_times.append(toc - tic)

    def transform_point_cloud(self, Y=None):
        if Y is None:
            self.TY = self.Y + np.dot(self.G, self.W)
            return
        else:
            return Y + np.dot(self.G, self.W)

    def update_variance(self):
        qprev = self.sigma2

        xPx      = np.dot(np.transpose(self.Pt1), np.sum(np.multiply(self.X, self.X), axis=1))
        yPy      = np.dot(np.transpose(self.P1),  np.sum(np.multiply(self.TY, self.TY), axis=1))
        trPXY    = np.sum(np.multiply(self.TY, np.dot(self.P, self.X)))

        self.sigma2 = (xPx - 2 * trPXY + yPy) / (self.Np * self.D)

        if self.sigma2 <= 0:
            self.sigma2 = self.tolerance / 10
        self.err = np.abs(self.sigma2 - qprev)

    def get_registration_parameters(self):
        return self.G, self.W
