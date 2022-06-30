import time

import numpy as np
from numpy.testing import assert_almost_equal, assert_array_almost_equal

import cython_functions as cy


class expectation_maximization_registration(object):
    def __init__(
        self,
        X,
        Y,
        max_iterations=100,
        tolerance=1e-5,
        w=0.0,
        verbose=True,
        print_reg_params=True,
        *args,
        **kwargs
    ):
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
        self.time_to_initialize_registration = 0
        self.time_to_initiate_registration = 0

        self.X = X
        self.Y = Y
        self.TY = None
        self.N = self.X.shape[0]
        self.D = self.X.shape[1]
        self.M = self.Y.shape[0]
        self.tolerance = tolerance
        self.w = w
        self.max_iterations = max_iterations
        self.iteration = 0
        self.err = self.tolerance + 1
        self.P = np.zeros((self.M, self.N))
        self.Pt1 = np.zeros((self.N,))
        self.P1 = np.zeros((self.M,))
        self.PX = np.zeros((self.M, self.D))
        self.Np = 0.0
        self.E = 0.0
        self.E_old = 0.0
        self.sigma2 = 0.0
        self.sigma2_prev = 0.0
        self.verbose = verbose
        self.print_reg_params = print_reg_params

    def register(self, callback=lambda **kwargs: None):
        tic = time.time()
        self.transform_point_cloud()
        self.sigma2 = cy.initialize_sigma2(self.X, self.Y)
        toc = time.time()
        self.time_to_initiate_registration = toc - tic
        while (
            (self.iteration < self.max_iterations)
            and (self.err > self.tolerance)
            and (self.sigma2 > np.finfo(float).eps)
        ):
            self.iterate()
            if callable(callback):
                kwargs = {"iteration": self.iteration, "error": self.err, "X": self.X, "Y": self.TY}
                callback(**kwargs)

        if self.print_reg_params is True:
            print("=" * 72)
            print("Registration Performance Metrics")
            print("=" * 72)
            print("Time to initialize EM: {}".format(self.time_to_initiate_registration))
            print(
                "Time to initialize registration: {}".format(self.time_to_initialize_registration)
            )
            print(
                "Average Expectation Time:                {:10.4f} +/- {:5.4f}".format(
                    np.mean(self.expectation_times), np.std(self.expectation_times)
                )
            )
            print(
                "Average Maximization Time:               {:10.4f} +/- {:5.4f}".format(
                    np.mean(self.maximization_times), np.std(self.maximization_times)
                )
            )
            print("Maximization Times - Per individual step")
            print(
                "Average Update Transform Time:           {:10.4f} +/- {:5.4f}".format(
                    np.mean(self.update_transform_times), np.std(self.update_transform_times)
                )
            )
            print(
                "Average Transform Time:                  {:10.4f} +/- {:5.4f}".format(
                    np.mean(self.transform_times), np.std(self.transform_times)
                )
            )
            print(
                "Average Update Variance Time:            {:10.4f} +/- {:5.4f}".format(
                    np.mean(self.update_variance_times), np.std(self.update_variance_times)
                )
            )
            print("")
            print("Number of iterations performed:          {}".format(self.iteration))
            print("Error at time of finish:                 {}".format(self.err))

        return self.TY, self.get_registration_parameters()

    def get_registration_parameters(self):
        raise NotImplementedError("Registration parameters should be defined in child classes.")

    def iterate(self):
        self.E_old = self.E  # E = negative log-likelihood function
        tic = time.time()
        self.expectation()
        toc = time.time()
        self.expectation_times.append(toc - tic)

        tic = time.time()
        self.maximization()
        toc = time.time()
        self.maximization_times.append(toc - tic)

        self.iteration += 1

        if self.verbose is True:
            while type(self.sigma2) is np.ndarray:
                self.sigma2 = self.sigma2[0]

            print("Iteration:{}".format(self.iteration))
            print(
                "ML:{:10.3f}; \t"
                "ML change (error):{:10.3f}; \t"
                "Sigma^2:{:10.3f}; \t"
                "Sigma^2 change:{:10.3f}".format(
                    self.E, self.err, self.sigma2, self.sigma2_prev - self.sigma2
                )
            )
            percent_done = int(72 * self.iteration / self.max_iterations)
            print("[" + "=" * percent_done + " " * (72 - percent_done) + "]")

    def expectation(self):
        self.P1, self.Pt1, self.PX, self.Np, self.E = cy.expectation_2(
            self.X, self.TY, self.sigma2, self.M, self.N, self.D, self.w
        )

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
