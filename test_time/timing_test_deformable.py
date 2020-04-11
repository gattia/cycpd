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
    print('Registration Error: {}'.format(reg.err))
    print('Number of iterations: {}'.format(reg.iteration))

def test_3D():
    X = np.loadtxt('../data/surface_points_bone_2_rigid_register_to_1_5k_points.npy')

    # Below are points from a completely different knee that were already rigidly registered to X
    # If there isnt something to make them "somewhat" close to one another, then the registration fails.
    # Therefore, this first step was performed to improve testing.
    Y = np.loadtxt('../data/surface_points_bone_1_5k_points.npy')

    # These will not perfectly align and they will not even be "done" when we get to iteration 100.
    # But this is a good starting point test.

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

    print('Registration Error: {}'.format(reg.err))
    print('Number of iterations: {}'.format(reg.iteration))
    # print(reg.sigma2)
    # print(reg.Np)



if __name__ == "__main__":
    test_2D()
    test_3D()
