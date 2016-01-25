# -*- coding: utf-8 -*-
"""
Created on Sun May 10 16:00:42 2015

@author: sc1
"""

from scipy.stats import beta
import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(1, 1)

a, b = 1, 50
mean, var, skew, kurt = beta.stats(a, b, moments='mvsk')

x = np.linspace(beta.ppf(0.01, a, b),
              beta.ppf(0.99, a, b), 100)

rv = beta(a, b)
ax.plot(x, rv.pdf(x), 'k-', lw=2, label='frozen pdf')

plt.show()