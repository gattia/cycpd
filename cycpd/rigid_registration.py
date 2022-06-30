import time
from builtins import super

import numpy as np
from numpy.testing import assert_almost_equal, assert_array_almost_equal

from .expectation_maximization_registration import expectation_maximization_registration


class rigid_registration(expectation_maximization_registration):
    """
    Rigid registration class.

    Parameters
    ----------
    R : array_like, optional
        The rotation matrix.

    t : array_like, optional
        The translation vector.

    s : array_like, optional
        The scale factor.

    scale : bool, optional
        Whether to lean scale for the point cloud.
    """

    def __init__(self, R=None, t=None, s=None, scale=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        tic = time.time()
        if self.D != 2 and self.D != 3:
            message = (
                "Rigid registration only supports 2D or 3D point clouds. Instead got {}.".format(
                    self.D
                )
            )
            raise ValueError(message)
        if s == 0:
            raise ValueError("A zero scale factor is not supported.")
        self.R = np.eye(self.D) if R is None else R
        self.t = np.atleast_2d(np.zeros((1, self.D))) if t is None else t
        self.s = 1.0 if s is None else s
        self.muX = None
        self.muY = None
        self.myU = None
        self.S = None
        self.C = None
        self.scale = scale

        toc = time.time()
        self.time_to_initialize_registration = toc - tic

    def update_transform(self):
        """
        Update the transform parameters.
        """
        # Replaced the PyCPD muX and muY because couldnt calcl myY the same (no self.P)
        # and wanted to use the same method for both.
        self.muX = np.divide(np.matmul(self.X.T, self.Pt1), self.Np)
        self.muY = np.divide(
            np.matmul(self.Y.T, self.P1), self.Np
        )  # changed previous code from PyCPD because we no longer store self.P
        A = np.matmul(self.PX.T, self.Y) - self.Np * np.matmul(
            self.muX[:, None], self.muY[:, None].T
        )  # A= X'P'*Y-X'P'1*1'P'Y/Np;

        U, self.S, V = np.linalg.svd(
            A, full_matrices=True
        )  # matlab and numpy return V in different orientations.
        self.S = np.diag(self.S)
        self.C = np.eye(self.D)
        # The following line means that we allow reflections. Can make this an option. See line 96:
        # https://github.com/markeroon/matlab-computer-vision-routines/blob/master/third_party/CoherentPointDrift/core/Rigid/cpd_rigid.m#L96
        self.C[self.D - 1, self.D - 1] = np.linalg.det(np.dot(U, V))

        self.R = np.transpose(np.dot(np.dot(U, self.C), V))
        if self.scale is True:
            self.s = np.trace(np.matmul(self.S, self.C)) / (
                np.sum(np.square(self.Y) * self.P1[:, None])
                - self.Np * (np.matmul(self.muY.T, self.muY))
            )
        elif self.scale is False:
            self.s = 1
        self.t = self.muX - self.s * np.dot(self.R.T, self.muY)

        # Use E (matlab function cpd_P) or L (matlab code for registration) to calculate the "error".
        # This is instead of using sigma which was the approach used by PyCPD. Doing this to ensure consistency between
        # This and the original implementation.
        # self.err = abs((self.E - self.E_old) / self.E)
        self.err = abs(
            self.E - self.E_old
        )  # The above is a "normalized" alternative that was used in the Matlab code.

    def transform_point_cloud(self, Y=None):
        """
        Transform a point cloud.

        Parameters
        ----------
        Y : array_like, optional
            The point cloud to transform.

        """
        if Y is None:
            self.TY = self.s * np.dot(self.Y, self.R) + self.t
            return
        else:
            return self.s * np.dot(Y, self.R) + self.t

    def update_variance(self):
        """
        Update the variance.
        """
        self.sigma2_prev = self.sigma2
        # A, B, C were just implemented to make the code reasier to read/interpret.
        # The code for self.sigma2 matches what is written in the matlab rigid example.
        A = np.sum(np.square(self.X) * self.Pt1[:, None])
        B = self.Np * np.matmul(self.muX[:, None].T, self.muX[:, None])

        if self.scale is True:
            C = self.s * np.trace(np.matmul(self.S, self.C))
            self.sigma2 = abs(A - B - C) / (self.Np * self.D)
        elif self.scale is False:
            C = np.sum(np.square(self.Y) * self.P1[:, None])
            D = self.Np * np.matmul(self.muY[:, None].T, self.muY[:, None])
            E = 2.0 * np.trace(np.matmul(self.S, self.C))
            self.sigma2 = abs(A - B + C - D - E) / (self.Np * self.D)
        # Still retaining the replacement/update of sigma2 if <=0 that was in PyCPD. This might not be be
        # necessary any more.
        if self.sigma2 <= 0:
            self.sigma2 = self.tolerance / 10

    def get_registration_parameters(self):
        """
        Get the registration parameters.

        Returns
        -------
        self.s : float
            The scale factor.

        self.R : array_like
            The rotation matrix.

        self.t : array_like
            The translation vector.
        """
        return self.s, self.R, self.t
