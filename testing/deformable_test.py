import pytest
import numpy as np
from numpy.testing import assert_almost_equal, assert_array_almost_equal
from pycpd import gaussian_kernel, deformable_registration

def test_2D():
    X = np.loadtxt('data/fish_target.txt')
    Y = np.loadtxt('data/fish_source.txt')

    reg = deformable_registration(**{ 'X': X, 'Y': Y })
    TY, _ = reg.register()
    assert_array_almost_equal(X, TY, decimal=1)

def test_3D():
    fish_target = np.loadtxt('data/fish_target.txt')
    X1 = np.zeros((fish_target.shape[0], fish_target.shape[1] + 1))
    X1[:,:-1] = fish_target
    X2 = np.ones((fish_target.shape[0], fish_target.shape[1] + 1))
    X2[:,:-1] = fish_target
    X = np.vstack((X1, X2))

    fish_source = np.loadtxt('data/fish_source.txt')
    Y1 = np.zeros((fish_source.shape[0], fish_source.shape[1] + 1))
    Y1[:,:-1] = fish_source
    Y2 = np.ones((fish_source.shape[0], fish_source.shape[1] + 1))
    Y2[:,:-1] = fish_source
    Y = np.vstack((Y1, Y2))

    reg = deformable_registration(**{ 'X': X, 'Y':Y })
    TY, _ = reg.register()
    assert_array_almost_equal(TY, X, decimal=0)
