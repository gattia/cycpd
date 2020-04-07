import numpy as np
import time
from cycpd.cython_functions import *

class expectation_maximization_registration(object):

    def __init__(self, X, Y, max_iterations=100, tolerance=0.001, w=0., *args, **kwargs):
        if type(X) is not np.ndarray or X.ndim != 2:
            raise ValueError("The target point cloud (X) must be at a 2D numpy array.")
        if type(Y) is not np.ndarray or Y.ndim != 2:
            raise ValueError("The source point cloud (Y) must be a 2D numpy array.")
        if X.shape[1] != Y.shape[1]:
            raise ValueError("Both point clouds need to have the same number of dimensions.")

        self.expectation_times = []
        self.maximization_times = []
        self.update_transform_times = []
        self.transform_times = []
        self.update_variance_times = []
        self.A_times = []
        self.B_times = []
        self.solve_times = []
        self.time_to_initialize_deformable_registration = 0
        self.time_to_initiate_registration = 0

        self.X              = X
        self.Y              = Y
        self.N              = self.X.shape[0]
        self.D              = self.X.shape[1]
        self.M              = self.Y.shape[0]
        self.tolerance      = tolerance
        self.w              = w
        self.max_iterations = max_iterations
        self.iteration      = 0
        self.err            = self.tolerance + 1
        self.P              = np.zeros((self.M, self.N))
        self.Pt1            = np.zeros((self.N, ))
        self.P1             = np.zeros((self.M, ))
        self.Np             = 0.


    def register(self, callback=lambda **kwargs: None):
        tic = time.time()
        self.transform_point_cloud()
        self.sigma2 = initialize_sigma2(self.X, self.TY)
        self.q = -self.err - self.N * self.D/2 * np.log(self.sigma2)
        toc = time.time()
        self.time_to_initiate_registration = toc-tic
        while self.iteration < self.max_iterations and self.err > self.tolerance:
            self.iterate()
            if callable(callback):
                kwargs = {'iteration': self.iteration, 'error': self.err, 'X': self.X, 'Y': self.TY}
                callback(**kwargs)
        print('Time to initialize EM registration: {}'.format(self.time_to_initiate_registration))
        print('Time to initialize Deformable registration: {}'.format(self.time_to_initialize_deformable_registration))
        print('Average Expectation Times: {} +/- {}'.format(np.mean(self.expectation_times),
                                                            np.std(self.expectation_times)))
        print('Average Maximization Times: {} +/- {}'.format(np.mean(self.maximization_times),
                                                             np.std(self.maximization_times)))
        print('')
        print('Average Update Transform Times: {} +/- {}'.format(np.mean(self.update_transform_times),
                                                            np.std(self.update_transform_times)))
        print('Average Transform Times: {} +/- {}'.format(np.mean(self.transform_times),
                                                             np.std(self.transform_times)))
        print('Average Update Variance Times: {} +/- {}'.format(np.mean(self.update_variance_times),
                                                          np.std(self.update_variance_times)))
        print('')
        if self.low_rank is False:
            print('Average A Times: {} +/- {}'.format(np.mean(self.A_times),
                                                                     np.std(self.A_times)))
            print('Average B Times: {} +/- {}'.format(np.mean(self.B_times),
                                                              np.std(self.B_times)))
            print('Average Solve Times: {} +/- {}'.format(np.mean(self.solve_times),
                                                                    np.std(self.solve_times)))
        print('Number of iterations used: {}'.format(self.iteration))
        print('Error at time of finish: {}'.format(self.err))

        return self.TY, self.get_registration_parameters()

    def get_registration_parameters(self):
        raise NotImplementedError("Registration parameters should be defined in child classes.")

    def iterate(self):
        tic = time.time()
        self.expectation()
        toc = time.time()
        self.expectation_times.append(toc-tic)
        tic = time.time()
        self.maximization()
        toc = time.time()
        self.maximization_times.append(toc-tic)
        self.iteration += 1

    def expectation(self):

        self.P, self.Pt1, self.P1, self.Np = expectation(self.X,
                                                         self.TY,
                                                         self.sigma2,
                                                         self.M,
                                                         self.N,
                                                         self.D,
                                                         self.w)

    def maximization(self):
        tic = time.time()
        self.update_transform()
        toc = time.time()
        self.update_transform_times.append(toc - tic)
        tic = time.time()
        self.transform_point_cloud()
        toc = time.time()
        self.transform_times.append(toc - tic)
        tic = time.time()
        self.update_variance()
        toc = time.time()
        self.update_variance_times.append(toc - tic)

        # print(self.P.shape)
        # print(self.P1.shape)
        # print(self.Pt1.shape)
        # print(self.Np)



# cdef class expectation_maximization_registration(object):
#     cdef double[:, :] X
#     cdef double[:, :] Y
#     cdef double[:, :] P
#     cdef double[:] Pt1
#     cdef double[:] P1
#     cdef double Np
#
#     cdef double sigma2, tolerance, w, err
#     cdef int max_iterations, iteration, N, M, D, _
#
#     def __init__(self, X, Y, max_iterations=100, tolerance=0.001, w=0., *args, **kwargs):
#         if type(X) is not np.ndarray or X.ndim != 2:
#             raise ValueError("The target point cloud (X) must be at a 2D numpy array.")
#         if type(Y) is not np.ndarray or Y.ndim != 2:
#             raise ValueError("The source point cloud (Y) must be a 2D numpy array.")
#         if X.shape[1] != Y.shape[1]:
#             raise ValueError("Both point clouds need to have the same number of dimensions.")
#
#         self.
#
#         self.X              = X
#         self.Y              = Y
#         self.N              = self.X.shape[0]
#         self.D              = self.X.shape[1]
#         self.M              = self.Y.shape[0]
#         self.tolerance      = tolerance
#         self.w              = w
#         self.max_iterations = max_iterations
#         self.iteration      = 0
#         self.err            = self.tolerance + 1
#         self.P              = np.zeros((self.M, self.N))
#         self.Pt1            = np.zeros((self.N, ))
#         self.P1             = np.zeros((self.M, ))
#         self.Np             = 0.
#
#     def register(self, callback=lambda **kwargs: None):
#         self.transform_point_cloud()
#         self.sigma2 = initialize_sigma2(self.X, self.TY)
#         self.q = -self.err - self.N * self.D/2 * np.log(self.sigma2)
#         while self.iteration < self.max_iterations and self.err > self.tolerance:
#             self.iterate()
#             if callable(callback):
#                 kwargs = {'iteration': self.iteration, 'error': self.err, 'X': self.X, 'Y': self.TY}
#                 callback(**kwargs)
#
#         return self.TY, self.get_registration_parameters()
#
#     def get_registration_parameters(self):
#         raise NotImplementedError("Registration parameters should be defined in child classes.")
#
#     def iterate(self):
#         tic = time.time()
#         self.expectation()
#         toc = time.time()
#         self.
#         self.maximization()
#         self.iteration += 1
#
#     def expectation(self):
#
#         P = np.sum((self.X[None,:,:] - self.TY[:,None,:])**2, axis=2)
#
#         c = (2 * np.pi * self.sigma2) ** (self.D / 2)
#         c = c * self.w / (1 - self.w)
#         c = c * self.M / self.N
#
#         P = np.exp(-P / (2 * self.sigma2))
#
#         den = np.sum(P, axis=0)
#         # den = np.tile(den, (self.M, 1))
#         den[den==0] = np.finfo(float).eps
#         den += c
#
#         self.P   = np.divide(P, den[None,:])
#         self.Pt1 = np.sum(self.P, axis=0)
#         self.P1  = np.sum(self.P, axis=1)
#         self.Np  = np.sum(self.P1)
#
#     def maximization(self):
#         self.update_transform()
#         self.transform_point_cloud()
#         self.update_variance()
