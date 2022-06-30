import argparse
from functools import partial

import matplotlib.pyplot as plt
import numpy as np

from cycpd import rigid_registration


def visualize(iteration, error, X, Y, ax, fig, tilt=25, rotation_factor=5, save_fig=False):
    plt.cla()
    ax[0].cla()
    ax[1].cla()

    ax[0].scatter(X[:, 0], X[:, 1], X[:, 2], color="red", label="Target", alpha=0.5, s=0.5)
    ax[0].scatter(Y[:, 0], Y[:, 1], Y[:, 2], color="blue", label="Source", alpha=0.8, s=0.5)
    ax[0].legend(loc="upper left", fontsize="x-large")
    ax[0].text2D(
        0.87,
        0.92,
        "Iteration: {:d}\nError: {:06.4f}".format(iteration, error),
        horizontalalignment="center",
        verticalalignment="center",
        transform=ax[0].transAxes,
        fontsize="x-large",
    )
    ax[0].view_init(tilt, 225)

    ax[1].scatter(X[:, 0], X[:, 1], X[:, 2], color="red", label="Target", alpha=0.5, s=0.5)
    ax[1].scatter(Y[:, 0], Y[:, 1], Y[:, 2], color="blue", label="Source", alpha=0.8, s=0.5)
    ax[1].view_init(tilt, rotation_factor * iteration)

    plt.draw()
    if save_fig is True:
        fig.savefig("rigid_{:04}.tiff".format(iteration))  # Used for making gif.

    plt.pause(0.001)


def main(save=False):
    theta = np.pi / 6.0
    R = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
    t = np.array([0.5, 1.0, -2.0])

    Y = np.loadtxt("../data/surface_points_bone_1_5k_points.npy")
    # the below line will let testing of registering two bones (the same) with a known rotation and
    # translation between them.
    # X = np.dot(Y, R) + t
    ## The below line will let testing of registration using two different bones.
    X = np.loadtxt("../data/surface_points_bone_2_5k_points.npy")

    fig = plt.figure(figsize=plt.figaspect(0.5))
    ax1 = fig.add_subplot(121, projection="3d")
    ax2 = fig.add_subplot(122, projection="3d")
    ax = [ax1, ax2]
    callback = partial(visualize, ax=ax, fig=fig, save_fig=save)

    reg = rigid_registration(**{"X": X, "Y": Y})
    # reg = rigid_registration(**{'X': X, 'Y': Y, 'scale': False})
    # The above shows an example where we dont "test" or determine the scale.
    # This makes it clear the CPD first actually shrinks the mesh and then "grows" it iteratively to make it
    # best fit the data.
    reg.register(callback)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rigid registration example")
    parser.add_argument(
        "-s",
        "--save",
        type=bool,
        nargs="+",
        default=False,
        help="True or False - to save figures of the example for a GIF etc.",
    )
    args = parser.parse_args()
    print(args)

    main(**vars(args))
