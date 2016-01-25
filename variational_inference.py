#!/usr/bin/env python
"""
A mean field variational inference algorithm where the assumption is that the proposal distribution can be written
as q(x) = q1(z1)q2(z2)...qn(zn)
"""
__author__ = 'benchamberlain'


import numpy as np
import matplotlib.pyplot as plt

# initialise all qi

# define the model which give p(X,Z)

# for each component of q, qj, we optimise with respect to all other components

# calculate the expectation of p(X,Z) wrt qi for all i except j

def sample_data():
    pass


def compute_lower_bound():
    pass


def compute_expectations():
    pass


def perform_maximisation():
    pass


def plot_countours():
    pass


def calculate_entropy():
    pass