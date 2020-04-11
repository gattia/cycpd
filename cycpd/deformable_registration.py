from builtins import super
import numpy as np
import time
from .expectation_maximization_registration import expectation_maximization_registration

def gaussian_kernel(Y, beta):
    diff = Y[None,:,:] - Y[:,None,:]
    diff = diff**2
    diff = np.sum(diff, 2)
    return np.exp(-diff / (2 * beta**2))

def lowrankQS(Y, beta, num_eig, eig_fgt=False):
    # M, D = Y.shape
    # hsigma=np.sqrt(2)*beta

    # if we do not use FGT we construct affinity matrix G and find the
    # first eigenvectors/values directly

    if eig_fgt is False:
        G=gaussian_kernel(Y, beta)
        S, Q=np.linalg.eigh(G)
        eig_indices = list(np.argsort(np.abs(S))[::-1][:num_eig])
        Q = Q[:,eig_indices] # eigenvectors
        S = S[eig_indices] # eigenvalues.

        return Q, S

    elif eig_fgt is True:
        raise Exception('Fast Gauss Transform Not Implemented!')

class deformable_registration(expectation_maximization_registration):
    def __init__(self, alpha=2, beta=2, low_rank=True, num_eig=100, eig_fgt=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        tic = time.time()
        self.alpha         = 2 if alpha is None else alpha
        self.beta          = 2 if beta is None else beta
        self.num_eig       = num_eig
        self.eigfgt        = eig_fgt
        self.W             = np.zeros((self.M, self.D))
        self.G             = gaussian_kernel(self.Y, self.beta)
        self.low_rank      = low_rank

        if self.low_rank is True:
            self.Q, self.S  = lowrankQS(self.Y, self.beta, self.num_eig, eig_fgt=eig_fgt)
            self.inv_S = np.diag(1./self.S)
            self.S = np.diag(self.S)
        toc = time.time()
        self.time_to_initialize_registration = toc - tic

    def update_transform(self):
        if self.low_rank is False:
            tic = time.time()
            A = np.dot(np.diag(self.P1), self.G) + self.alpha * self.sigma2 * np.eye(self.M)
            toc = time.time()
            self.A_times.append(toc-tic)
            tic = time.time()
            B = self.PX - np.dot(np.diag(self.P1), self.Y)
            toc = time.time()
            self.B_times.append(toc - tic)
            tic = time.time()
            self.W = np.linalg.solve(A, B)
            toc = time.time()
            self.solve_times.append(toc - tic)
        else:
            dP = np.diag(self.P1)
            dPQ = np.matmul(dP, self.Q)
            F = self.PX - np.matmul(dP, self.Y)

            self.W = 1 / (self.alpha * self.sigma2) * (F - np.matmul(dPQ, (
                np.linalg.solve((self.alpha * self.sigma2 * self.inv_S + np.matmul(self.Q.T, dPQ)), (np.matmul(self.Q.T, F))))))
            QtW = np.matmul(self.Q.T, self.W)
            self.E = self.E + self.alpha/2 * np.trace(np.matmul(QtW.T, np.matmul(self.S, QtW)))
            # self.err = abs((self.E - self.E_old) / self.E)
            self.err = abs(self.E - self.E_old)
            # The absolute difference is more conservative (does more iterations) than the line above it which
            # is calculating the normalized change in the E(L). This calculation was changed to match the matlab
            # code created for low_rank matrices.

    def transform_point_cloud(self, Y=None):
        if self.low_rank is False:
            if Y is None:
                self.TY = self.Y + np.dot(self.G, self.W)
                return
            else:
                return Y + np.dot(self.G, self.W)
        elif self.low_rank is True:
            if Y is None:
                self.TY = self.Y + np.matmul(self.Q, np.matmul(self.S, np.matmul(self.Q.T, self.W)))
                return
            else:
                return Y + np.matmul(self.Q, np.matmul(self.S, np.matmul(self.Q.T, self.W)))

    def update_variance(self):
        self.sigma2_prev = self.sigma2
        A = np.sum(np.square(self.X) * self.Pt1[:, None])
        B = np.sum(np.square(self.TY) * self.P1[:, None])
        C = 2 * np.trace(np.matmul(self.PX.T, self.TY))
        self.sigma2 = abs( A + B - C ) / (self.Np * self.D)

        if self.sigma2 <= 0:
            self.sigma2 = self.tolerance / 10

        # self.err = abs(self.sigma2_prev - self.sigma2)

    def get_registration_parameters(self):
        return self.G, self.W
