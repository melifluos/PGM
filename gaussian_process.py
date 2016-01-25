"""
Gaussian process test module
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

__author__ = 'benchamberlain'


def cov_func(data, alpha=1, l=2):
    """
    Calculates the covariance matrix
    :param data: The training data
    :param alpha: The variance
    :param l:
    :return: A numpy array covariance matrix
    """
    size = len(data)
    tiled_data = np.tile(data, (size, 1))
    input_mat = tiled_data - tiled_data.T
    input_mat = np.power(input_mat, 2 * np.ones(input_mat.shape)) / (2.0 * np.power(l, 2))
    cov_mat = alpha * np.exp(-1 * input_mat)
    return cov_mat


def test():
    data = np.array([-3, 1.2, 1.4])
    mat = gaussian_cov_function(data)
    print mat
    fig, ax = plt.subplots()
    heatmap = ax.pcolor(mat, cmap=plt.cm.Blues)
    # plt.plot([1,2,4],[1,3,5])
    plt.show()
    # plt.plot(data)
    print 'done'


class GP:
    def __init__(self):
        pass

    def mean(self, data):
        """
        Calculate the mean
        :param data: The input values
        :return:
        """
        return 0.25 * (data ** 2)

    def covariance(self, data, l=1, alpha=1):
        """
        Calculates the covariance matrix
        :param data: The training data
        :param alpha: The variance
        :param l:
        :return: A numpy array covariance matrix
        """
        size = len(data)
        tiled_data = np.tile(data, (size, 1))
        input_mat = tiled_data - tiled_data.T
        input_mat = np.power(input_mat, 2 * np.ones(input_mat.shape)) / (2.0 * np.power(l, 2))
        cov_mat = alpha * np.exp(-1 * input_mat)
        return cov_mat


def sample_gp(GP, sample_points, n_samples=20):
    """
    To visualise a GP you must take samples as it is infinite. To do this you choose a set of sample points
    and evaluate the mean and covariance functions at these points. This gives a mean and covariance vector,
    and each sample from that is viewed as a function.
    :param GP: A Gaussian Process object
    :param sample_points: A set of points to evaluate the mean and covariance functions at
    :param n_samples: The number of times to sample the multivariate Gaussian / number of functions to generate
    :return:
    """
    X = sample_points[:, np.newaxis]
    mean = GP.mean(X)
    cov = GP.covariance(X)
    Z = np.random.multivariate_normal(mean, cov, size=n_samples)
    return Z


def plot_GP(X, f):
    """
    Plot the GP curves
    :param X: The sample points
    :param f: The function values at these points
    :return:
    """
    plt.figure()
    for i in range(f.shape[0]):
        plt.plot(X[:], Z[i, :])


def plot_test():
    x = np.arange(0.0, 2, 0.01)
    y1 = np.sin(2 * np.pi * x)
    y2 = 1.2 * np.sin(4 * np.pi * x)

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)

    ax1.fill_between(x, 0, y1, facecolor='grey', alpha=0.5)
    ax1.set_ylabel('between y1 and 0')

    ax2.fill_between(x, y1, 1)
    ax2.set_ylabel('between y1 and 1')

    ax3.fill_between(x, y1, y2)
    ax3.set_ylabel('between y1 and y2')
    ax3.set_xlabel('x')

    plt.show()


def plot_func(data, noise=10e-6):
    fig, ax = plt.subplots()
    means = mean_func(data)
    covs = cov_func(data)
    # This is the standard way of generating samples from a multivariate normal distribution
    R = np.linalg.cholesky(covs + np.diag(np.ones(len(data))) * noise)
    y_min = means - 2 * np.random.standard_normal(size=(1, len(data))).dot(R)
    y_max = means + 2 * np.random.standard_normal(size=(1, len(data))).dot(R)
    y = means + np.random.standard_normal(size=(1, len(data))).dot(R)
    print y_min.shape, y_max.shape
    plot_data = data[np.newaxis, :]
    print plot_data.shape
    ax.fill_between(plot_data, y_min, y_max, where=y_max > y_min, facecolor='green')
    ax.plot(plot_data, y.T)
    plt.show()

def run_stuff():
    data = np.linspace(0., 1., 500)
    gp = GP()
    samples = sample_gp(gp, data)
    plot_GP(data[:, np.newaxis], samples)


if __name__ == '__main__':
    data = np.linspace(0., 1., 500)
    gp = GP()
    samples = sample_gp(gp, data)
    plot_GP(data[:, np.newaxis], samples)
    # plot_func(data)
    # plot_test()
