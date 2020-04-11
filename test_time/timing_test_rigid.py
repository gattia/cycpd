import numpy as np
from numpy.testing import assert_almost_equal, assert_array_almost_equal
from cycpd import rigid_registration
import time


def test_2D():
    tic = time.time()
    theta = np.pi / 6.0
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    t = np.array([0.5, 1.0])

    Y = np.loadtxt('../data/fish_target.txt')
    X = np.dot(Y, R) + np.tile(t, (np.shape(Y)[0], 1))

    reg = rigid_registration(**{ 'X': X, 'Y':Y })
    reg.low_rank = True
    TY, (s_reg, R_reg, t_reg) = reg.register()
    assert_almost_equal(1.0, s_reg)
    assert_array_almost_equal(R, R_reg)
    assert_array_almost_equal(t, t_reg)
    assert_array_almost_equal(X, TY)

    toc = time.time()
    print('Test 2D Rigid took on fish took: {}'.format(toc - tic))


def test_3D():
    tic = time.time()
    theta = np.pi / 6.0
    R = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
    t = np.array([0.5, 1.0, -2.0])

    Y = np.loadtxt('../data/surface_points_bone_1_5k_points.npy')
    # Y = np.zeros((knee_target.shape[0], knee_target.shape[1] + 1))
    # Y[:,:-1] = knee_target
    X = np.dot(Y, R) + np.tile(t, (np.shape(Y)[0], 1))

    reg = rigid_registration(**{ 'X': X, 'Y':Y })
    reg.low_rank = True
    TY, (s_reg, R_reg, t_reg) = reg.register()
    assert_almost_equal(1.0, s_reg)
    assert_array_almost_equal(R, R_reg)
    assert_array_almost_equal(t, t_reg)
    assert_array_almost_equal(X, TY)

    toc = time.time()
    print('Test 3D Rigid took on knee with 5k points took: {}'.format(toc - tic))




if __name__ == "__main__":
    test_2D()
    test_3D()
