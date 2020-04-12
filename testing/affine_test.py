import pytest
import numpy as np
from numpy.testing import assert_almost_equal, assert_array_almost_equal
from cycpd import affine_registration


def test_2d(timing=False,
            verbose=False,
            print_reg_params=False):
    if timing is True:
        tic = time.time()
    B = np.array([[1.0, 0.5], [0, 1.0]])
    t = np.array([0.5, 1.0])

    try:
        Y = np.loadtxt('data/fish_target.txt')
    except OSError:
        Y = np.loadtxt('../data/fish_target.txt')
    X = np.dot(Y, B) + np.tile(t, (np.shape(Y)[0], 1))

    reg = affine_registration(**{'X': X, 'Y': Y,
                                 'verbose': verbose,
                                 'print_reg_params': print_reg_params})
    TY, (B_reg, t_reg) = reg.register()
    assert_array_almost_equal(B, B_reg.T)
    assert_array_almost_equal(t, t_reg)
    assert_array_almost_equal(X, TY)

    if timing is True:
        toc = time.time()
        print('Test 2D Affine took on fish took: {}'.format(toc - tic))


def test_3d(timing=False,
            verbose=False,
            print_reg_params=False):
    if timing is True:
        tic = time.time()
    B = np.array([[1.0, 0.5, 0.0], [0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    t = np.array([0.5, 1.0, -2.0])
    try:
        Y = np.loadtxt('../data/surface_points_bone_1_5k_points.npy')
    except OSError:
        Y = np.loadtxt('data/surface_points_bone_1_5k_points.npy')
    X = np.dot(Y, B) + np.tile(t, (np.shape(Y)[0], 1))

    reg = affine_registration(**{'X': X, 'Y': Y,
                                 'verbose': verbose,
                                 'print_reg_params': print_reg_params})
    TY, (B_reg, t_reg) = reg.register()
    assert_array_almost_equal(B, B_reg.T)
    assert_array_almost_equal(t, t_reg)
    assert_array_almost_equal(X, TY)

    if timing is True:
        toc = time.time()
        print('Test 3D Affine on 5k point knee took: {}'.format(toc - tic))


if __name__ == "__main__":
    import time
    test_2d(timing=True, verbose=True, print_reg_params=True)
    test_3d(timing=True, verbose=True, print_reg_params=True)
