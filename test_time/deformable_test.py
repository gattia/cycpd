import pytest
import numpy as np
from numpy.testing import assert_almost_equal, assert_array_almost_equal
from cycpd import gaussian_kernel, deformable_registration
import time

def test_2D():
    X = np.loadtxt('../data/fish_target.txt')
    Y = np.loadtxt('../data/fish_source.txt')

    tic = time.time()
    reg = deformable_registration(**{ 'X': X, 'Y': Y })
    TY, _ = reg.register()
    toc = time.time()
    assert_array_almost_equal(X, TY, decimal=1)
    print('Test 2D Deformable took on fish took: {}'.format(toc - tic))

def test_3D():
    X = np.loadtxt('../data/surface_points_bone_1_5k_points.npy')
    # fish_target = np.loadtxt('../data/fish_target.txt')
    # X1 = np.zeros((fish_target.shape[0], fish_target.shape[1] + 1))
    # X1[:,:-1] = fish_target
    # X2 = np.ones((fish_target.shape[0], fish_target.shape[1] + 1))
    # X2[:,:-1] = fish_target
    # X = np.vstack((X1, X2))

    Y = np.loadtxt('../data/surface_points_bone_2_5k_points.npy')
    # fish_source = np.loadtxt('../data/fish_source.txt')
    # Y1 = np.zeros((fish_source.shape[0], fish_source.shape[1] + 1))
    # Y1[:,:-1] = fish_source
    # Y2 = np.ones((fish_source.shape[0], fish_source.shape[1] + 1))
    # Y2[:,:-1] = fish_source
    # Y = np.vstack((Y1, Y2))

    tic = time.time()
    reg = deformable_registration(**{ 'X': X, 'Y':Y })
    toc = time.time()
    time_setup_deformable = toc - tic
    tic = time.time()
    TY, _ = reg.register()
    # assert_array_almost_equal(TY, X, decimal=0)
    toc = time.time()
    time_do_registration = toc - tic
    print('Test 3D Deformable setup registration time: {}'.format(time_setup_deformable))
    print('Test 3D Deformable do registration time: {}'.format(time_do_registration))
    print('Test 3D Deformable took on fish took: {}'.format(time_do_registration + time_setup_deformable))



if __name__ == "__main__":
    test_2D()
    test_3D()
