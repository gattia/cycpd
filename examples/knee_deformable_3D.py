from functools import partial
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from cycpd import deformable_registration, rigid_registration
import numpy as np
import time

def visualize(iteration, error, X, Y, ax, fig, tilt=0, rotation_factor=5):
    plt.cla()
    ax[0].cla()
    ax[1].cla()

    ax[0].scatter(X[:, 0], X[:, 1], X[:, 2], color='red', label='Target', alpha=0.5, s=0.5)
    ax[0].scatter(Y[:, 0], Y[:, 1], Y[:, 2], color='blue', label='Source', alpha=0.8, s=0.5)
    ax[0].legend(loc='upper left', fontsize='x-large')
    ax[0].text2D(0.87, 0.92, 'Iteration: {:d}\nError: {:06.4f}'.format(iteration, error),
                 horizontalalignment='center', verticalalignment='center', transform=ax[0].transAxes,
                 fontsize='x-large')
    ax[0].view_init(tilt, 40)

    ax[1].scatter(X[:, 0], X[:, 1], X[:, 2], color='red', label='Target', alpha=0.5, s=0.5)
    ax[1].scatter(Y[:, 0], Y[:, 1], Y[:, 2], color='blue', label='Source', alpha=0.8, s=0.5)
    ax[1].view_init(tilt, rotation_factor * iteration)

    plt.draw()
    # fig.savefig('deformable_{:04}.tiff'.format(iteration))  # Used for making gif.
    plt.pause(0.001)

    # plt.cla()
    # ax.scatter(X[:,0],  X[:,1], X[:,2], color='red', label='Target', alpha=0.5, s=0.5)
    # ax.scatter(Y[:,0],  Y[:,1], Y[:,2], color='blue', label='Source', alpha=0.8, s=0.5)
    # ax.text2D(0.87, 0.92, 'Iteration: {:d}\nError: {:06.4f}'.format(iteration, error), horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize='x-large')
    # ax.legend(loc='upper left', fontsize='x-large')
    # plt.draw()
    # plt.pause(1.0)

def main():
    X = np.loadtxt('../data/surface_points_bone_deformable_target.npy')

    # Below are points from a completely different knee that were already rigidly registered to X
    # If there isnt something to make them "somewhat" close to one another, then the registration fails.
    # Therefore, this first step was performed to improve testing.
    Y = np.loadtxt('../data/surface_points_bone_1_5k_points.npy')

    # These will not perfectly align and they will not even be "done" when we get to iteration 100.
    # But this is a good starting point test and shows the movement of one of the meshes over time as it tries to align
    # with the other mesh.

    fig = plt.figure(figsize=plt.figaspect(0.5))
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')
    ax = [ax1, ax2]
    callback = partial(visualize, ax=ax, fig=fig)

    reg = deformable_registration(**{ 'X': X,
                                      'Y': Y,
                                      'max_iterations': 100,
                                      'alpha': 0.1,
                                      'beta': 3
                                      })
    TY, _ = reg.register(callback)

    plt.show()

    differences = X[:, None, :] - TY[None, :, :]
    distances = np.sqrt(np.sum(differences ** 2, axis=2))
    min_x_dist_per_ty_point = np.min(distances, axis=0)
    sorted_distances = np.sort(min_x_dist_per_ty_point)
    worst_one_percent_error = sorted_distances[int(len(min_x_dist_per_ty_point) * 0.99)]
    worst_five_percent_error = sorted_distances[int(len(min_x_dist_per_ty_point) * 0.95)]
    worst_ten_percent_error = sorted_distances[int(len(min_x_dist_per_ty_point) * 0.90)]

    plt.figure()
    plt.hist(sorted_distances)
    plt.show()

    print('Bottom 1% error: {}'.format(worst_one_percent_error))
    print('Bottom 5% error: {}'.format(worst_five_percent_error))
    print('Bottom 10% error: {}'.format(worst_ten_percent_error))

if __name__ == '__main__':
    main()
