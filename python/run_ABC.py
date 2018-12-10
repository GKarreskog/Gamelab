# Load functions
%run vectorized_funs.py
%run abc_funs.py
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# ABC stuff
from pyabc.visualization import plot_kde_matrix
from pyabc.visualization import plot_kde_1d

import os
import tempfile

import scipy.stats as st
import scipy as scp


from pyabc import (ABCSMC, RV, Distribution,
                   PercentileDistanceFunction)

import pyabc

# %% Load data
with open('data_dicts.pkl', 'rb') as input:
    load_dict = pickle.load(input)
    init_strats_dict = load_dict["init_strats_dict"]
    init_p1_dict = load_dict["init_p1_dict"]
    init_p2_dict = load_dict["init_p2_dict"]
    payoffs_dict = load_dict["payoffs_dict"]
    role_dict = load_dict["role_dict"]
    id_dict = load_dict["id_dict"]
    actual_plays_dict = load_dict["actual_plays_dict"]

for gid in payoffs_dict:
    payoffs_dict[gid] = [payoffs_dict[gid][gid-1][i].astype(float) for i in range(2)]


actuals_dict = dict()

for gid in actual_plays_dict:
    actuals_dict[gid] = flatten_single_hist(actual_plays_dict[gid])

# %% Set parameters
data_dict = actuals_dict
# gids = [2,3,4,5,6]
gids = [1,2,3]
n_runs = 1
bw = 0.05
p1_size = 20
p2_size = 20
rounds = 30
games = payoffs_dict
n_particles = 1000
max_pops = 10
min_accept_rate = 0.0001
init_ε = 10
α = 0.5

# %% Run analysis
plt.switch_backend("Agg")
flat_hists, shape = flatten_data([np.array([data_dict[gid]]) for gid in gids])
y = {"data":flat_hists, "shape":shape}
