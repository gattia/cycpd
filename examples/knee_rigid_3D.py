from functools import partial
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from cycpd import rigid_registration
import numpy as np
import time

def visualize(iteration, error, X, Y, ax, tilt=25, rotation_factor=5):
    plt.cla()
    ax[0].cla()
    ax[1].cla()

    ax[0].scatter(X[:, 0], X[:, 1], X[:, 2], color='red', label='Target', alpha=0.5, s=0.5)
    ax[0].scatter(Y[:, 0], Y[:, 1], Y[:, 2], color='blue', label='Source', alpha=0.8, s=0.5)
    ax[0].legend(loc='upper left', fontsize='x-large')
    ax[0].text2D(0.87, 0.92, 'Iteration: {:d}\nError: {:06.4f}'.format(iteration, error),
                 horizontalalignment='center', verticalalignment='center', transform=ax[0].transAxes,
                 fontsize='x-large')
    ax[0].view_init(tilt, 225)

    ax[1].scatter(X[:, 0], X[:, 1], X[:, 2], color='red', label='Target', alpha=0.5, s=0.5)
    ax[1].scatter(Y[:, 0], Y[:, 1], Y[:, 2], color='blue', label='Source', alpha=0.8, s=0.5)
    ax[1].view_init(tilt, rotation_factor * iteration)

    plt.draw()
    plt.pause(0.001)

    # plt.cla()
    # ax.scatter(X[:,0],  X[:,1], X[:,2], color='red', label='Target', alpha=0.5, s=0.5)
    # ax.scatter(Y[:,0],  Y[:,1], Y[:,2], color='blue', label='Source', alpha=0.8, s=0.5)
    # ax.text2D(0.87, 0.92, 'Iteration: {:d}\nError: {:06.4f}'.format(iteration, error), horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize='x-large')
    # ax.legend(loc='upper left', fontsize='x-large')
    # plt.draw()
    # plt.pause(0.001)

def main():
    theta = np.pi / 6.0
    R = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
    t = np.array([0.5, 1.0, -2.0])

    Y = np.loadtxt('../data/surface_points_bone_1_5k_points.npy')
    # Y = np.zeros((knee_target.shape[0], knee_target.shape[1] + 1))
    # Y[:,:-1] = knee_target
    X = np.dot(Y, R) + t

    fig = plt.figure(figsize=plt.figaspect(0.5))
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')
    ax = [ax1, ax2]
    callback = partial(visualize, ax=ax)

    reg = rigid_registration(**{ 'X': X, 'Y':Y})
    # reg = rigid_registration(**{'X': X, 'Y': Y, 'scale': False})
    # The above shows an example where we dont "test" or determine the scale.
    # This makes it clear the CPD first actually shrinks the mesh and then "grows" it iteratively to make it
    # best fit the data.
    reg.register(callback)
    plt.show()

if __name__ == '__main__':
    main()
