from builtins import super
import numpy as np
import time
from .expectation_maximization_registration import expectation_maximization_registration


class affine_registration(expectation_maximization_registration):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        tic = time.time()
        self.B = np.eye(self.D)
        self.t = np.zeros(self.D)
        self.B1 = None
        self.B2 = None

        toc = time.time()
        self.time_to_initialize_registration = toc - tic

    def update_transform(self):
        self.muX = np.divide(np.matmul(self.X.T, self.Pt1), self.Np)
        self.muY = np.divide(
            np.matmul(self.Y.T, self.P1), self.Np
        )  # changed previous code from PyCPD because we no longer store self.P

        self.B1 = np.matmul(self.PX.T, self.Y) - self.Np * np.matmul(
            self.muX[:, None], self.muY[:, None].T
        )
        self.B2 = np.matmul((self.Y * self.P1[:, None]).T, self.Y) - self.Np * np.matmul(
            self.muY[:, None], self.muY[:, None].T
        )
        self.B = np.linalg.solve(self.B2.T, self.B1.T).T

        self.t = np.squeeze(self.muX[:, None] - np.matmul(self.B, self.muY[:, None]))

        self.err = abs(self.E - self.E_old)

    def transform_point_cloud(self, Y=None):
        if Y is None:
            self.TY = np.dot(self.Y, self.B.T) + self.t
            return
        else:
            return np.dot(Y, self.B.T) + self.t

    def update_variance(self):
        self.sigma2_prev = self.sigma2
        A = np.sum(np.square(self.X) * self.Pt1[:, None])
        B = self.Np * np.matmul(self.muX[:, None].T, self.muX[:, None])
        C = np.trace(np.matmul(self.B1, self.B.T))
        self.sigma2 = abs(A - B - C) / (self.Np * self.D)

        if self.sigma2 <= 0:
            self.sigma2 = self.tolerance / 10

    def get_registration_parameters(self):
        return self.B, self.t
