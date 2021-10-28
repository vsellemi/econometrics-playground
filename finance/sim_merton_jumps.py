#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simulating Merton Model With Jumps
@author: victorsellemi
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

# simulation parameters
S0 = 100
r  = .05
sigma = .2
lamb  = .75
mu    = -.6
delta = .25
T = 1
M = 200
I = 10**4
dt = T/M

# simulating training data
rj = lamb * (np.exp(mu+.5*delta**2) - 1)
S  = np.zeros((M+1,I))
S[0] = S0
z1 = np.random.standard_normal((M+1,I))
z2 = np.random.standard_normal((M+1,I))
y  = np.random.poisson(lamb*dt, (M+1,I)) # lists one for jumps

for t in range(1,M+1):
    S[t] = S[t-1] * (np.exp((r - rj - .5*sigma**2)*dt + sigma*np.sqrt(dt)*z1[t]) + \
        (np.exp(mu+delta*z2[t]) - 1)*y[t]) 
    S[t] = np.maximum(S[t],0)

# visualize example process
plt.plot(S[:,0], linewidth = .7)
