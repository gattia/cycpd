import pytest
import numpy as np
from numpy.testing import assert_almost_equal, assert_array_almost_equal
from cycpd import gaussian_kernel, deformable_registration

def test_2D():
    try:
        X = np.loadtxt('data/fish_target.txt')
    except:
        X = np.loadtxt('../data/fish_target.txt')
    try:
        Y = np.loadtxt('data/fish_source.txt')
    except:
        Y = np.loadtxt('../data/fish_source.txt')

    reg = deformable_registration(**{'X': X, 'Y': Y, 'verbose': False})
    TY, _ = reg.register()
    assert_array_almost_equal(X, TY, decimal=1)

def test_3D(one_percent_error=1.2, five_percent_error=0.9, ten_percent_error=0.6):
    try:
        X = np.loadtxt('../data/surface_points_bone_deformable_target.npy')
    except:
        X = np.loadtxt('data/surface_points_bone_deformable_target.npy')

    # Below are points from a completely different knee that were already rigidly registered to X
    # If there isnt something to make them "somewhat" close to one another, then the registration fails.
    # Therefore, this first step was performed to improve testing.
    try:
        Y = np.loadtxt('../data/surface_points_bone_1_5k_points.npy')
    except:
        Y = np.loadtxt('data/surface_points_bone_1_5k_points.npy')

    # These will not perfectly align and they will not even be "done" when we get to iteration 100.
    # But this is a good starting point test.
    reg = deformable_registration(**{'X': X,
                                     'Y': Y,
                                     'max_iterations': 100,
                                     'alpha': 0.1,
                                     'beta': 3,
                                     'verbose': False})
    TY, _ = reg.register()

    differences = X[:, None, :] - TY[None, :, :]
    distances = np.sqrt(np.sum(differences ** 2, axis=2))
    min_x_dist_per_ty_point = np.min(distances, axis=0)
    sorted_distances = np.sort(min_x_dist_per_ty_point)
    worst_one_percent_error = sorted_distances[int(len(min_x_dist_per_ty_point) * 0.99)]
    worst_five_percent_error = sorted_distances[int(len(min_x_dist_per_ty_point) * 0.95)]
    worst_ten_percent_error = sorted_distances[int(len(min_x_dist_per_ty_point) * 0.90)]

    print('Worst 1% error: {}'.format(worst_one_percent_error))
    print('Worst 5% error: {}'.format(worst_five_percent_error))
    print('Worst 10% error: {}'.format(worst_ten_percent_error))

    assert worst_one_percent_error < one_percent_error
    assert worst_five_percent_error < five_percent_error
    assert worst_ten_percent_error < ten_percent_error



