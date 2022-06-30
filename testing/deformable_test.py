import os

import numpy as np
from numpy.testing import assert_array_almost_equal

from cycpd import deformable_registration

dir_path = os.path.dirname(os.path.realpath(__file__))


def test_2d(timing=False, verbose=False, print_reg_params=False):
    if timing is True:
        tic = time.time()
    try:
        X = np.loadtxt(os.path.join(dir_path, "..", "data", "fish_target.txt"))
    except OSError:
        raise Exception("Error finding data!")
    try:
        Y = np.loadtxt(os.path.join(dir_path, "..", "data", "fish_source.txt"))
    except OSError:
        raise Exception("Error finding data!")

    reg = deformable_registration(
        **{
            "X": X,
            "Y": Y,
            "verbose": verbose,
            "print_reg_params": print_reg_params,
            "max_iterations": 500,
        }
    )
    TY, _ = reg.register()
    assert_array_almost_equal(X, TY, decimal=1)

    if timing is True:
        toc = time.time()
        print("Test 2D Affine took on fish took: {}".format(toc - tic))


def test_3d(
    timing=False,
    print_errors=False,
    verbose=False,
    print_reg_params=False,
    one_percent_error=0.5,
    five_percent_error=0.4,
    ten_percent_error=0.35,
):
    if timing is True:
        tic = time.time()
    try:
        X = np.loadtxt(
            os.path.join(dir_path, "..", "data", "surface_points_bone_deformable_target.npy")
        )
    except OSError:
        raise Exception("Error finding data!")
    # Below are points from a completely different knee that were already rigidly registered to X
    # If there isnt something to make them "somewhat" close to one another, then the registration fails.
    # Therefore, this first step was performed to improve testing.
    try:
        Y = np.loadtxt(os.path.join(dir_path, "..", "data", "surface_points_bone_1_5k_points.npy"))
    except OSError:
        raise Exception("Error finding data!")

    # These will not perfectly align and they will not even be "done" when we get to iteration 100.
    # But this is a good starting point test.
    reg = deformable_registration(
        **{
            "X": X,
            "Y": Y,
            "max_iterations": 500,
            "alpha": 0.1,
            "beta": 3,
            "verbose": verbose,
            "print_reg_params": print_reg_params,
        }
    )
    TY, _ = reg.register()

    differences = X[:, None, :] - TY[None, :, :]
    distances = np.sqrt(np.sum(differences**2, axis=2))
    min_x_dist_per_ty_point = np.min(distances, axis=0)
    sorted_distances = np.sort(min_x_dist_per_ty_point)
    worst_one_percent_error = sorted_distances[int(len(min_x_dist_per_ty_point) * 0.99)]
    worst_five_percent_error = sorted_distances[int(len(min_x_dist_per_ty_point) * 0.95)]
    worst_ten_percent_error = sorted_distances[int(len(min_x_dist_per_ty_point) * 0.90)]

    if print_errors is True:
        print("Worst 1% error: {}".format(worst_one_percent_error))
        print("Worst 5% error: {}".format(worst_five_percent_error))
        print("Worst 10% error: {}".format(worst_ten_percent_error))

    assert worst_one_percent_error < one_percent_error
    assert worst_five_percent_error < five_percent_error
    assert worst_ten_percent_error < ten_percent_error

    if timing is True:
        toc = time.time()
        print("Test 3D Deformable on knee with 5k points took: {}".format(toc - tic))


if __name__ == "__main__":
    import time

    test_2d(
        timing=True,
        verbose=True,
        print_reg_params=True,
    )
    test_3d(timing=True, verbose=True, print_reg_params=True, print_errors=True)
