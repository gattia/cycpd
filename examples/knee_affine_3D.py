from functools import partial
import matplotlib.pyplot as plt
from cycpd import affine_registration
import numpy as np
import argparse


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
    # fig.savefig('affine_{:04}.tiff'.format(iteration)) # Used for making gif.
    plt.pause(0.001)


def main(save=False):
    theta = np.pi / 6.0
    Y = np.loadtxt("../data/surface_points_bone_1_5k_points.npy")
    Hxy = 0
    Hxz = 0
    Hyx = 0
    Hyz = 0
    Hzx = 0
    Hzy = 0
    shear_matrix = [[1, Hxy, Hxz], [Hyx, 1, Hyz], [Hzx, Hzy, 1]]
    R = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
    t = np.array([0.5, 1.0, -2.0])

    R = np.matmul(R, shear_matrix)

    X = np.dot(Y, R) + t

    fig = plt.figure(figsize=plt.figaspect(0.5))
    ax1 = fig.add_subplot(121, projection="3d")
    ax2 = fig.add_subplot(122, projection="3d")
    ax = [ax1, ax2]
    callback = partial(visualize, ax=ax, fig=fig, save_fig=save)

    reg = affine_registration(**{"X": X, "Y": Y})
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
