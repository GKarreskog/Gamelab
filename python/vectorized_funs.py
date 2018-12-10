### Load modules and games
import numpy as np
import copy 
import sys 
import warnings
import scipy.stats as scst 
import string 
import pickle as pickle
import random
import multiprocessing as mp
import pandas as pd
import time
from sklearn.neighbors import KernelDensity
from IPython import get_ipython
# from general_funs import get_toulouse_games
from math import exp



ipython = get_ipython()

import numba
from numba import jit, guvectorize, vectorize, float64, prange         # import the decorator
from numba import int32, float32    # import the types
# %load_ext line_profiler



# games = get_toulouse_games()
# for gid in games:
#     games[gid][0] = games[gid][0].astype(float)
#     games[gid][1] = games[gid][1].astype(float)

# np.set_printoptions(precision=3, suppress=True)

### jit funcitons

factorials = np.array([1, 1, 2, 6, 24, 120, 720, 5040, 40320,
    362880, 3628800, 39916800, 479001600,
    6227020800, 87178291200, 1307674368000,
    20922789888000, 355687428096000, 6402373705728000,
    121645100408832000, 2432902008176640000])

@jit(nopython=True,parallel=False)
def poisson_p(i,τ):
    return (τ**i * np.exp(-τ))/factorials[i]

@jit(nopython=True,parallel=False)
def weighted_rand_int(weigths):
    p = np.random.rand()
    for i in range(len(weigths)):
        if weigths[i] > p:
            return i
        p = p - weigths[i]
    else:
        return len(weigths)

@jit(nopython=True,parallel=False)
def poisson_weight(poisson_vec, strats, j):
    strat = np.zeros_like(strats[0])
    poisson_sum = 0
    for i in range(j):
        strat += poisson_vec[i]*strats[i]
        poisson_sum += poisson_vec[i]
    strat = strat/poisson_sum
    return strat


@jit(nopython=True,parallel=False)
def init_LPCHM(self_payoffs, opp_payoffs, params, pure=True, k_rand=True):
    τ = params[0]
    λ = params[1]
    self_n = self_payoffs.shape[0]
    opp_n = opp_payoffs.shape[0]
    if k_rand:
        k = int(np.random.poisson(τ))
        while k > 10:
            k = int(np.random.poisson(τ))
    else:
        k = int(τ)
   
    self_s = np.ones((k+1, self_n))/(self_n)
    opp_s = np.ones((k+1, opp_n))/(opp_n)
    poisson_weights_all = np.array([poisson_p(i,τ) for i in range(k+1)])
    if k == 0:
        return self_s[0]
    else: 
        for j in range(1, k+1):
            opp_s_guess = poisson_weight(poisson_weights_all, opp_s, j)
            self_s_guess = poisson_weight(poisson_weights_all, self_s, j)
            best_reply_logit(self_payoffs, opp_s_guess, λ, self_s[j], pure=True)
            best_reply_logit(opp_payoffs, self_s_guess, λ, opp_s[j], pure=True)
        return self_s[-1]

@jit(nopython=True, parallel=False)
def init_LPCHM_for(strats, self_payoffs, opp_payoffs, params, pure=True, k_rand=True):
    for i in prange(len(strats)):
        strats[i][:] = init_LPCHM(self_payoffs, opp_payoffs, params, pure=pure, k_rand=k_rand)[:]

@jit(nopython=True, parallel=False)
def best_reply_logit(payoff_mat, opp_s, λ, strat,  pure=False):
# def best_reply_logit(payoff_mat, opp_s, λ, pure=True):
    n_strats = payoff_mat.shape[0]
    avg_payoff = payoff_mat @ opp_s
    new_strats = np.exp(λ*avg_payoff - (λ*avg_payoff).max())
    new_strats = new_strats/new_strats.sum()
    if pure:
        choice = weighted_rand_int(new_strats)
        new_strats = np.zeros(n_strats)
        new_strats[choice] = 1.
    strat[:] = new_strats[:]

@jit(nopython=True)
def sample_simplex(strats):
    sample = np.random.rand(len(strats) + 1)
    sample[0] = 0
    sample[-1] = 1.
    sample.sort()
    sample = sample[1:] - sample[0:-1]
    strats[:] = sample[:]


@jit(nopython=True)
def best_reply(self_strats, payoff_mat, opp_s):
    n_strats = payoff_mat.shape[0]
    avg_payoff = payoff_mat @ opp_s
    best_rep = np.zeros(n_strats)
    if (avg_payoff == avg_payoff.max()).sum() == 1:
        best_rep[avg_payoff == avg_payoff.max()] = 1
    else: 
        sample = np.zeros((avg_payoff == avg_payoff.max()).sum())
        sample_simplex(sample)
        best_rep[avg_payoff == avg_payoff.max()] = sample
    # best_rep[avg_payoff == avg_payoff.max()] = 1
    # best_rep = best_rep/best_rep.sum()
    self_strats[:] = best_rep[:]

@jit(nopython=True)
def best_reply_for(self_strats, payoff_mat, opp_s, params):
    n_strats = payoff_mat.shape[0]
    for i in range(len(self_strats)):
        if params[i][0] > np.random.rand():
            if params[i][1] < np.random.rand():
                # avg_payoff = payoff_mat @ opp_s
                # best_rep = np.zeros(n_strats)
                # best_rep[avg_payoff == avg_payoff.max()] = 1
                # best_rep = best_rep/best_rep.sum()
                # self_strats[i][:] = best_rep[:]
                best_reply(self_strats[i], payoff_mat, opp_s)
            else:
                rand_rep = np.zeros(n_strats)
                rand_rep[np.random.randint(n_strats)] = 1.
                self_strats[i][:] = rand_rep[:]

@jit(nopython=True)
def LBR_reply_for(self_strats, payoff_mat, opp_s, params):
    n_strats = payoff_mat.shape[0]
    for i in range(len(self_strats)):
        p = params[i][0]
        # ε = params[i][1]
        λ = params[i][1]
        if p > np.random.rand():
            # if ε < np.random.rand():
            best_reply_logit(payoff_mat, opp_s, λ, self_strats[i], pure=True)
            # else:
            #     rand_rep = np.zeros(n_strats)
            #     rand_rep[np.random.randint(n_strats)] = 1.
            #     self_strats[i][:] = rand_rep[:]

@jit(nopython=True)
def LPCHM_reply_for(self_strats, self_payoff_mat, opp_payoff_mat, self_s, opp_s, params):
    n_strats = self_payoff_mat.shape[0]
    opp_n_strats = opp_payoff_mat.shape[0]
    for i in range(len(self_strats)):
        # p = params[i][0]
        τ = params[i][0]
        λ = params[i][1]
        # if p > np.random.rand():
        k = int(np.random.poisson(τ))
        while k > 10:
            k = int(np.random.poisson(τ))
        if k == 0:
            # self_strats[i][:] = self_s[:]
            self_strats[i][:] = self_strats[i][:]
        else:
            self_s_vec = np.ones((k+1, n_strats))/(n_strats)
            opp_strats_vec = np.ones((k+1, opp_n_strats))/(opp_n_strats)
            self_s_vec[0][:] = self_s[:]
            opp_strats_vec[0][:] = opp_s[:]
            poisson_weights_all = np.array([poisson_p(i,τ) for i in range(k+1)])
            for j in range(1, k+1):
                opp_s_guess = poisson_weight(poisson_weights_all, opp_strats_vec, j)
                self_s_guess = poisson_weight(poisson_weights_all, self_s_vec, j)
                if j < k: 
                    best_reply(self_s_vec[j], self_payoff_mat, opp_s_guess)
                    best_reply(opp_strats_vec[j], opp_payoff_mat, self_s_guess)
                else:
                    best_reply_logit(self_payoff_mat, opp_s_guess, λ, self_s_vec[j], pure=True)
                    best_reply_logit(opp_payoff_mat, self_s_guess, λ, opp_strats_vec[j], pure=True)
            self_strats[i][:] = self_s_vec[-1][:] 

@jit(nopython=True)
def PCHM_reply_for(self_strats, self_payoff_mat, opp_payoff_mat, self_s, opp_s, params):
    n_strats = self_payoff_mat.shape[0]
    opp_n_strats = opp_payoff_mat.shape[0]
    for i in range(len(self_strats)):
        p = params[i][0]
        τ = params[i][1]
        if p > np.random.rand():
            k = int(np.random.poisson(τ)) 
            while k > 10:
                k = int(np.random.poisson(τ))
            if k == 0:
                self_strats[i][:] = self_s[:]
            else:
                self_s_vec = np.ones((k+1, n_strats))/(n_strats)
                opp_strats_vec = np.ones((k+1, opp_n_strats))/(opp_n_strats)
                self_s_vec[0][:] = self_s[:]
                opp_strats_vec[0][:] = opp_s[:]
                poisson_weights_all = np.array([poisson_p(i,τ) for i in range(k+1)])
                for j in range(1, k+1):
                    opp_s_guess = poisson_weight(poisson_weights_all, opp_strats_vec, j)
                    self_s_guess = poisson_weight(poisson_weights_all, self_s_vec, j)
                    best_reply(self_s_vec[j], self_payoff_mat, opp_s_guess)
                    best_reply(opp_strats_vec[j], opp_payoff_mat, self_s_guess)
                self_strats[i][:] = self_s_vec[-1][:]

@jit(nopython=True)
def JPCHM_reply_for(self_strats, self_payoff_mat, opp_payoff_mat, self_s, opp_s, params):
    n_strats = self_payoff_mat.shape[0]
    opp_n_strats = opp_payoff_mat.shape[0]
    for i in range(len(self_strats)):
        p = params[i][0]
        τ = params[i][1]
        
        if p > np.random.rand():
            k = int(np.random.poisson(τ))
            while k > 10:
                k = int(np.random.poisson(τ))
            if k == 0:
                rand_rep = np.zeros(n_strats)
                # rand_rep[np.random.randint(n_strats)] = 1.
                rand_rep[weighted_rand_int(self_s)] = 1.
                # self_strats[i][:] = rand_rep[:]
                self_strats[i][:] = rand_rep[:]
                
            else:
                self_s_vec = np.ones((k+1, n_strats))/(n_strats)
                opp_strats_vec = np.ones((k+1, opp_n_strats))/(opp_n_strats)
                self_s_vec[0][:] = self_s[:]
                opp_strats_vec[0][:] = opp_s[:]
                poisson_weights_all = np.array([poisson_p(i,τ) for i in range(k+1)])
                for j in range(1, k+1):
                    opp_s_guess = (1-p)*opp_s + p*poisson_weight(poisson_weights_all, opp_strats_vec, j)
                    self_s_guess = (1-p)*self_s + p*poisson_weight(poisson_weights_all, self_s_vec, j)
                    best_reply(self_s_vec[j], self_payoff_mat, opp_s_guess)
                    best_reply(opp_strats_vec[j], opp_payoff_mat, self_s_guess)
                self_strats[i][:] = self_s_vec[-1][:]

@jit(nopython=True)
def LJPCHM_reply_for(self_strats, self_payoff_mat, opp_payoff_mat, self_s, opp_s, params):
    n_strats = self_payoff_mat.shape[0]
    opp_n_strats = opp_payoff_mat.shape[0]
    for i in range(len(self_strats)):
        p = params[i][0]
        τ = params[i][1]
        λ = params[i][2]
        if p > np.random.rand():
            k = int(np.random.poisson(τ))
            while k > 10:
                k = int(np.random.poisson(τ))
            if k == 0:
                rand_rep = np.zeros(n_strats)
                # # rand_rep[np.random.randint(n_strats)] = 1.
                rand_rep[weighted_rand_int(self_s)] = 1.
                # # self_strats[i][:] = rand_rep[:]
                self_strats[i][:] = rand_rep[:]
                # self_strats[i][:] = self_s[:]
            else:
                self_s_vec = np.ones((k+1, n_strats))/(n_strats)
                opp_strats_vec = np.ones((k+1, opp_n_strats))/(opp_n_strats)
                self_s_vec[0][:] = self_s[:]
                opp_strats_vec[0][:] = opp_s[:]
                poisson_weights_all = np.array([poisson_p(i,τ) for i in range(k+1)])
                for j in range(1, k+1):
                    #TODO: Remove comment
                    opp_s_guess = (1-p)*opp_s + p*poisson_weight(poisson_weights_all, opp_strats_vec, j)
                    self_s_guess = (1-p)*self_s + p*poisson_weight(poisson_weights_all, self_s_vec, j)
                    if j < k:
                        best_reply(self_s_vec[j], self_payoff_mat, opp_s_guess)
                        best_reply(opp_strats_vec[j], opp_payoff_mat, self_s_guess)
                    else:
                        best_reply_logit(self_payoff_mat, opp_s_guess, λ, self_s_vec[j], pure=True)
                        best_reply_logit(opp_payoff_mat, self_s_guess, λ, opp_strats_vec[j], pure=True)
                self_strats[i][:] = self_s_vec[-1][:]

@jit(nopython=True)
def L1PCHM_reply_for(self_strats, self_payoff_mat, opp_payoff_mat, self_s, opp_s, params):
    n_strats = self_payoff_mat.shape[0]
    opp_n_strats = opp_payoff_mat.shape[0]
    for i in range(len(self_strats)):
        p = params[i][0]
        τ = params[i][1]
        λ = params[i][2]
        if p > np.random.rand():
            k = 1 + int(np.random.poisson(τ))
            while (k > 10) or (k==0):
                k = int(np.random.poisson(τ))
            if k == 0:
                rand_rep = np.zeros(n_strats)
                # # rand_rep[np.random.randint(n_strats)] = 1.
                rand_rep[weighted_rand_int(self_s)] = 1.
                # # self_strats[i][:] = rand_rep[:]
                self_strats[i][:] = rand_rep[:]
                # self_strats[i][:] = self_s[:]
            else:
                self_s_vec = np.ones((k+1, n_strats))/(n_strats)
                opp_strats_vec = np.ones((k+1, opp_n_strats))/(opp_n_strats)
                # self_s_vec[0][:] = self_s[:]
                # opp_strats_vec[0][:] = opp_s[:]
                poisson_weights_all = np.array([poisson_p(i,τ) for i in range(k+1)])
                for j in range(1, k+1):
                    if j == 1:
                        opp_s_guess = opp_s
                        self_s_guess = self_s
                    else:
                        opp_s_guess = (1-p)*opp_s + p*poisson_weight(poisson_weights_all, opp_strats_vec, j-1)
                        self_s_guess = (1-p)*self_s + p*poisson_weight(poisson_weights_all, self_s_vec, j-1)
                    if j < k:
                        best_reply(self_s_vec[j], self_payoff_mat, opp_s_guess)
                        best_reply(opp_strats_vec[j], opp_payoff_mat, self_s_guess)
                    else:
                        best_reply_logit(self_payoff_mat, opp_s_guess, λ, self_s_vec[j], pure=True)
                        best_reply_logit(opp_payoff_mat, self_s_guess, λ, opp_strats_vec[j], pure=True)
                self_strats[i][:] = self_s_vec[-1][:]




@jit(nopython=True)
def LK_reply_for(self_strats, self_payoff_mat, opp_payoff_mat, self_s, opp_s, params):
    n_strats = self_payoff_mat.shape[0]
    opp_n_strats = opp_payoff_mat.shape[0]
    old_self = np.zeros(n_strats)
    old_opp = np.zeros(opp_n_strats)
    self_best = np.zeros(n_strats)
    opp_best = np.zeros(opp_n_strats)
    for i in range(len(self_strats)):
        p = params[i][0]
        τ = params[i][1]
        if p > np.random.rand():
            k = int(np.random.poisson(τ))
            while k > 10:
                k = int(np.random.poisson(τ))
            if k == 0:
                self_strats[i][:] = self_s[:]
            else:
                self_best[:] = self_s[:]
                opp_best[:] = opp_s[:]
                for _ in range(0,k):
                    old_self[:] = self_best[:]
                    old_opp[:] = opp_best[:]
                    best_reply(self_best, self_payoff_mat, old_opp)
                    best_reply(opp_best, opp_payoff_mat, old_self)
                self_strats[i][:] = self_best[:]

@jit(nopython=True)
def LLK_reply_for(self_strats, self_payoff_mat, opp_payoff_mat, self_s, opp_s, params):
    n_strats = self_payoff_mat.shape[0]
    opp_n_strats = opp_payoff_mat.shape[0]
    old_self = np.zeros(n_strats)
    old_opp = np.zeros(opp_n_strats)
    self_best = np.zeros(n_strats)
    opp_best = np.zeros(opp_n_strats)
    for i in range(len(self_strats)):
        p = params[i][0]
        τ = params[i][1]
        λ = params[i][2]
        if p > np.random.rand():
            k = int(np.random.poisson(τ))
            while k > 10:
                k = int(np.random.poisson(τ))
            if k == 0:
                self_strats[i][:] = self_s[:]
            else:
                self_best[:] = self_s[:]
                opp_best[:] = opp_s[:]
                for l in range(1,k+1):
                    old_self[:] = self_best[:]
                    old_opp[:] = opp_best[:]
                    if l < k:
                        best_reply(self_best, self_payoff_mat, old_opp)
                        best_reply(opp_best, opp_payoff_mat, old_self)
                    else: 
                        best_reply_logit(self_best, self_payoff_mat, λ, old_opp, pure=False)
                        best_reply_logit(opp_best, opp_payoff_mat, λ, old_self, pure=False)
                self_strats[i][:] = self_best[:]

@jit(nopython=True)
def EWA_reply_for(self_strats, self_payoff_mat, self_As, self_Ns, opp_s, params):
    n_strats = self_payoff_mat.shape[0]
    avg_payoffs = self_payoff_mat @ opp_s
    for i in range(len(self_strats)):
        prev_payoff = np.dot(self_strats[i],avg_payoffs)
        p = params[i][0]
        λ = params[i][1]
        φ = params[i][2]
        ρ = params[i][3]
        δ = params[i][4]
        old_N = self_Ns[i]
        self_Ns[i] = 1 + self_Ns[i]*ρ
        for s in range(n_strats):
            # self_As[i][s] = (φ*old_N*self_As[i][s] + (δ + (1-δ)*self_strats[i][s])*avg_payoffs[s])/self_Ns[i]
            self_As[i][s] = (φ*old_N*self_As[i][s] + δ*avg_payoffs[s] + (1-δ)*self_strats[i][s]*prev_payoff)/self_Ns[i]
            if p > np.random.rand():
                new_strats = np.exp(λ*self_As[i] - (λ*self_As[i]).max())
                new_strats = new_strats/new_strats.sum()
                reply = np.zeros(n_strats)
                reply[weighted_rand_int(new_strats)] = 1.
                self_strats[i][:] = reply[:]


@jit(nopython=True)
def sample_normal_bounded(params_vec, μ_vec, σ_vec, lower_vec, upper_vec):
    for i in range(len(params_vec)):
        params_vec[i] = np.random.normal(μ_vec[i],σ_vec[i])
        while params_vec[i] < lower_vec[i] or params_vec[i] > upper_vec[i]:
            params_vec[i] = np.random.normal(μ_vec[i],σ_vec[i])

@jit(nopython=True)
def draw_beta_from_μσ(μ_in, σ_in, a, b):
    μ = (μ_in-a)/(b-a)
    σ = σ_in/(b-a)
    σ = min(σ, np.sqrt((1-μ)*μ) - 0.000001)
    α = ((1-μ)/σ**2 - 1/μ)*μ**2
    β = α*(1/μ - 1)
    return np.random.beta(α,β)*(b-a) + a

@jit(nopython=True)
def draw_beta(α,β, a, b):
    return np.random.beta(α,β)*(b-a) + a

@jit(nopython=True)
def sample_beta(params_vec, μ_vec, σ_vec, lower_vec, upper_vec):
    for i in range(len(params_vec)):
        params_vec[i] = draw_beta(μ_vec[i],σ_vec[i], lower_vec[i], upper_vec[i])


@jit(nopython=True)
def initiate_params(params, μ_vec, random=True, σ_vec=np.array([]), lower_vec=np.array([]), upper_vec=np.array([])):
    if not random:
        for i in range(len(params)):
            params[i][:] = μ_vec[:] 
    else:
        for i in range(len(params)):
            sample_normal_bounded(params[i], μ_vec, σ_vec, lower_vec, upper_vec)
            # sample_beta(params[i], μ_vec, σ_vec, lower_vec, upper_vec)


@jit
def calc_history(strats, hist):
    for i in range(len(hist)):
        hist[i] = np.mean(strats[:,i])

@jit(nopython=True)
def init_run(p1_strats, p2_strats, p1_params, p2_params, p1_history, p2_history, p1_payoffs, p2_payoffs, rounds, params_vec, init_params, σ_vec=np.array([]), lower_vec=np.array([]), upper_vec=np.array([]), random=True, actual_init=False):
    initiate_params(p1_params, params_vec, σ_vec=σ_vec, lower_vec=lower_vec, upper_vec=upper_vec, random=random)
    initiate_params(p2_params, params_vec, σ_vec=σ_vec, lower_vec=lower_vec, upper_vec=upper_vec, random=random)
    if not actual_init:
        init_LPCHM_for(p1_strats,p1_payoffs, p2_payoffs, init_params)
        init_LPCHM_for(p2_strats,p2_payoffs, p1_payoffs, init_params)
    calc_history(p1_strats, p1_history[0]) 
    calc_history(p2_strats, p2_history[0]) 

def init_As_and_Ns(self_payoff_mat, self_As, self_Ns):
    self_Ns[:] = 1.
    self_As = np.repeat([self_payoff_mat.mean(axis=1)],len(self_Ns),axis=0)



@jit(nopython=True)
def run_BR(p1_strats, p2_strats, p1_params, p2_params, p1_history, p2_history, p1_payoffs, p2_payoffs, rounds, params_vec, init_params, σ_vec=np.array([]), lower_vec=np.array([]), upper_vec=np.array([]), random=True, actual_init=False):
    init_run(p1_strats, p2_strats, p1_params, p2_params, p1_history, p2_history, p1_payoffs, p2_payoffs, rounds, params_vec, init_params, σ_vec=σ_vec, lower_vec=lower_vec, upper_vec=upper_vec, random=random, actual_init=actual_init)
    for i in range(1, rounds):
        best_reply_for(p1_strats, p1_payoffs, p2_history[i-1], p1_params)
        best_reply_for(p2_strats, p2_payoffs, p1_history[i-1], p2_params)
        calc_history(p1_strats, p1_history[i])
        calc_history(p2_strats, p2_history[i])

@jit(nopython=True)
def run_LBR(p1_strats, p2_strats, p1_params, p2_params, p1_history, p2_history, p1_payoffs, p2_payoffs, rounds, params_vec, init_params, σ_vec=np.array([]), lower_vec=np.array([]), upper_vec=np.array([]), random=True, actual_init=False):
    init_run(p1_strats, p2_strats, p1_params, p2_params, p1_history, p2_history, p1_payoffs, p2_payoffs, rounds, params_vec, init_params, σ_vec=σ_vec, lower_vec=lower_vec, upper_vec=upper_vec, random=random, actual_init=actual_init)
    for i in range(1, rounds):
        LBR_reply_for(p1_strats, p1_payoffs, p2_history[i-1], p1_params)
        LBR_reply_for(p2_strats, p2_payoffs, p1_history[i-1], p2_params)
        calc_history(p1_strats, p1_history[i])
        calc_history(p2_strats, p2_history[i])

@jit(nopython=True)
def run_LK(p1_strats, p2_strats, p1_params, p2_params, p1_history, p2_history, p1_payoffs, p2_payoffs, rounds, params_vec, init_params, σ_vec=np.array([]), lower_vec=np.array([]), upper_vec=np.array([]), random=True, actual_init=False):
    init_run(p1_strats, p2_strats, p1_params, p2_params, p1_history, p2_history, p1_payoffs, p2_payoffs, rounds, params_vec, init_params, σ_vec=σ_vec, lower_vec=lower_vec, upper_vec=upper_vec, random=random, actual_init=actual_init)
    for i in range(1, rounds):
        LK_reply_for(p1_strats, p1_payoffs, p2_payoffs, p1_history[i-1], p2_history[i-1], p1_params)
        LK_reply_for(p2_strats, p2_payoffs, p1_payoffs, p2_history[i-1], p1_history[i-1], p2_params)
        calc_history(p1_strats, p1_history[i])
        calc_history(p2_strats, p2_history[i])

@jit(nopython=True)
def run_LLK(p1_strats, p2_strats, p1_params, p2_params, p1_history, p2_history, p1_payoffs, p2_payoffs, rounds, params_vec, init_params, σ_vec=np.array([]), lower_vec=np.array([]), upper_vec=np.array([]), random=True, actual_init=False):
    init_run(p1_strats, p2_strats, p1_params, p2_params, p1_history, p2_history, p1_payoffs, p2_payoffs, rounds, params_vec, init_params, σ_vec=σ_vec, lower_vec=lower_vec, upper_vec=upper_vec, random=random, actual_init=actual_init)
    for i in range(1, rounds):
        LLK_reply_for(p1_strats, p1_payoffs, p2_payoffs, p1_history[i-1], p2_history[i-1], p1_params)
        LLK_reply_for(p2_strats, p2_payoffs, p1_payoffs, p2_history[i-1], p1_history[i-1], p2_params)
        calc_history(p1_strats, p1_history[i])
        calc_history(p2_strats, p2_history[i])

@jit(nopython=True)
def run_PCHM(p1_strats, p2_strats, p1_params, p2_params, p1_history, p2_history, p1_payoffs, p2_payoffs, rounds, params_vec, init_params, σ_vec=np.array([]), lower_vec=np.array([]), upper_vec=np.array([]), random=True, actual_init=False):
    init_run(p1_strats, p2_strats, p1_params, p2_params, p1_history, p2_history, p1_payoffs, p2_payoffs, rounds, params_vec, init_params, σ_vec=σ_vec, lower_vec=lower_vec, upper_vec=upper_vec, random=random, actual_init=actual_init)
    for i in range(1, rounds):
        PCHM_reply_for(p1_strats, p1_payoffs, p2_payoffs, p1_history[i-1], p2_history[i-1], p1_params)
        PCHM_reply_for(p2_strats, p2_payoffs, p1_payoffs, p2_history[i-1], p1_history[i-1], p2_params)
        calc_history(p1_strats, p1_history[i])
        calc_history(p2_strats, p2_history[i])
    
@jit(nopython=True)
def run_LPCHM(p1_strats, p2_strats, p1_params, p2_params, p1_history, p2_history, p1_payoffs, p2_payoffs, rounds, params_vec, init_params, σ_vec=np.array([]), lower_vec=np.array([]), upper_vec=np.array([]), random=True, actual_init=False):
    init_run(p1_strats, p2_strats, p1_params, p2_params, p1_history, p2_history, p1_payoffs, p2_payoffs, rounds, params_vec, init_params, σ_vec=σ_vec, lower_vec=lower_vec, upper_vec=upper_vec, random=random, actual_init=actual_init)
    for i in range(1, rounds):
        LPCHM_reply_for(p1_strats, p1_payoffs, p2_payoffs, p1_history[i-1], p2_history[i-1], p1_params)
        LPCHM_reply_for(p2_strats, p2_payoffs, p1_payoffs, p2_history[i-1], p1_history[i-1], p2_params)
        calc_history(p1_strats, p1_history[i])
        calc_history(p2_strats, p2_history[i])

@jit(nopython=True)
def run_JPCHM(p1_strats, p2_strats, p1_params, p2_params, p1_history, p2_history, p1_payoffs, p2_payoffs, rounds, params_vec, init_params, σ_vec=np.array([]), lower_vec=np.array([]), upper_vec=np.array([]), random=True, actual_init=False):
    init_run(p1_strats, p2_strats, p1_params, p2_params, p1_history, p2_history, p1_payoffs, p2_payoffs, rounds, params_vec, init_params, σ_vec=σ_vec, lower_vec=lower_vec, upper_vec=upper_vec, random=random, actual_init=actual_init)
    for i in range(1, rounds):
        JPCHM_reply_for(p1_strats, p1_payoffs, p2_payoffs, p1_history[i-1], p2_history[i-1], p1_params)
        JPCHM_reply_for(p2_strats, p2_payoffs, p1_payoffs, p2_history[i-1], p1_history[i-1], p2_params)
        calc_history(p1_strats, p1_history[i])
        calc_history(p2_strats, p2_history[i])

@jit(nopython=True)
def run_LJPCHM(p1_strats, p2_strats, p1_params, p2_params, p1_history, p2_history, p1_payoffs, p2_payoffs, rounds, params_vec, init_params, σ_vec=np.array([]), lower_vec=np.array([]), upper_vec=np.array([]), random=True, actual_init=False):
    init_run(p1_strats, p2_strats, p1_params, p2_params, p1_history, p2_history, p1_payoffs, p2_payoffs, rounds, params_vec, init_params, σ_vec=σ_vec, lower_vec=lower_vec, upper_vec=upper_vec, random=random, actual_init=actual_init)
    for i in range(1, rounds):
        LJPCHM_reply_for(p1_strats, p1_payoffs, p2_payoffs, p1_history[i-1], p2_history[i-1], p1_params)
        LJPCHM_reply_for(p2_strats, p2_payoffs, p1_payoffs, p2_history[i-1], p1_history[i-1], p2_params)
        calc_history(p1_strats, p1_history[i])
        calc_history(p2_strats, p2_history[i])

@jit(nopython=True)
def run_L1PCHM(p1_strats, p2_strats, p1_params, p2_params, p1_history, p2_history, p1_payoffs, p2_payoffs, rounds, params_vec, init_params, σ_vec=np.array([]), lower_vec=np.array([]), upper_vec=np.array([]), random=True, actual_init=False):
    init_run(p1_strats, p2_strats, p1_params, p2_params, p1_history, p2_history, p1_payoffs, p2_payoffs, rounds, params_vec, init_params, σ_vec=σ_vec, lower_vec=lower_vec, upper_vec=upper_vec, random=random, actual_init=actual_init)
    for i in range(1, rounds):
        L1PCHM_reply_for(p1_strats, p1_payoffs, p2_payoffs, p1_history[i-1], p2_history[i-1], p1_params)
        L1PCHM_reply_for(p2_strats, p2_payoffs, p1_payoffs, p2_history[i-1], p1_history[i-1], p2_params)
        calc_history(p1_strats, p1_history[i])
        calc_history(p2_strats, p2_history[i])

# @jit(nopython=True)
def run_EWA(p1_strats, p2_strats, p1_params, p2_params, p1_history, p2_history, p1_payoffs, p2_payoffs, p1_As, p2_As, p1_Ns, p2_Ns, rounds, params_vec, init_params, σ_vec=np.array([]), lower_vec=np.array([]), upper_vec=np.array([]), random=True, actual_init=False):
    init_run(p1_strats, p2_strats, p1_params, p2_params, p1_history, p2_history, p1_payoffs, p2_payoffs, rounds, params_vec, init_params, σ_vec=σ_vec, lower_vec=lower_vec, upper_vec=upper_vec, random=random, actual_init=actual_init)
    init_As_and_Ns(p1_payoffs, p1_As, p1_Ns)
    init_As_and_Ns(p2_payoffs, p2_As, p2_Ns)
    for i in range(1, rounds):
        EWA_reply_for(p1_strats, p1_payoffs, p1_As, p1_Ns, p2_history[i-1], p1_params)
        EWA_reply_for(p2_strats, p2_payoffs, p2_As, p2_Ns, p1_history[i-1], p2_params)
        calc_history(p1_strats, p1_history[i])
        calc_history(p2_strats, p2_history[i])

def create_h_vec(p1_hist, p2_hist, n):
    h_vec = [[p1_hist.copy(), p2_hist.copy()] for i in range(n)]
    return h_vec

@jit(nopython=True)
def flatten_h_jit(hists, flat_hists):
    rounds = len(hists[0][0])
    n_p1 = len(hists[0][0][0])
    n_p2 = len(hists[0][1][0])
    for h in range(len(hists)):
        for r in range(len(hists[0][0])):
            for s in range(len(hists[0][0][0])):
                flat_hists[h][r*n_p1 + s] = hists[h][0][r][s]
            for s in range(len(hists[0][1][0])):
                flat_hists[h][rounds*n_p1 + r*n_p2 + s] = hists[h][1][r][s]

### Population stuff
class Population: 
    """ A population consists of a number of agents who repeatedly plays a game """
    def __init__(self, p1_payoffs, p2_payoffs, rounds, n_p1, n_p2, params_vec=np.array([]), init_params=np.array([1.5,1.]), σ_vec=np.array([]), lower_vec=np.array([]), upper_vec=np.array([]), random_params=True, init_strats=False):
        self.p1_payoffs = p1_payoffs 
        self.p1_nstrats = p1_payoffs.shape[0]
        self.p2_payoffs = p2_payoffs 
        self.p2_nstrats = p2_payoffs.shape[0]
        self.rounds = rounds
        self.params_vec = params_vec
        self.σ_vec = σ_vec
        self.lower_vec = lower_vec 
        self.upper_vec = upper_vec
        self.random_params = random_params
        self.p1_params = np.zeros((n_p1, len(params_vec))) 
        initiate_params(self.p1_params, params_vec, σ_vec=σ_vec, lower_vec=lower_vec, upper_vec=upper_vec, random=random_params)
        self.p2_params = np.zeros((n_p2, len(params_vec))) 
        initiate_params(self.p2_params, params_vec,  σ_vec=σ_vec, lower_vec=lower_vec, upper_vec=upper_vec, random=random_params)
        self.init_params = init_params
        self.p1_strats = np.zeros((n_p1, self.p1_nstrats))
        self.p2_strats = np.zeros((n_p2, self.p2_nstrats))
        self.p1_history = np.zeros((rounds, self.p1_nstrats))
        self.p2_history = np.zeros((rounds, self.p2_nstrats))
        self.actual_init = False
        if init_strats:
            self.p1_strats = np.repeat([init_strats[0]], n_p1, axis=0)
            self.p2_strats = np.repeat([init_strats[1]], n_p2, axis=0)
            calc_history(self.p1_strats, self.p1_history[0])
            calc_history(self.p2_strats, self.p2_history[0])
            self.actual_init = True
        self.p1_As = np.repeat([p1_payoffs.mean(axis=1)],n_p1,axis=0)
        self.p2_As = np.repeat([p2_payoffs.mean(axis=1)],n_p1,axis=0)
        self.p1_Ns = np.ones(n_p1)
        self.p2_Ns = np.ones(n_p2)

    def run_BR(self):
        run_BR(self.p1_strats, self.p2_strats, self.p1_params, self.p2_params, self.p1_history, self.p2_history, self.p1_payoffs, self.p2_payoffs, self.rounds, self.params_vec, self.init_params, actual_init=self.actual_init, σ_vec=self.σ_vec, random=self.random_params, lower_vec=self.lower_vec, upper_vec=self.upper_vec)
        return [self.p1_history, self.p2_history]
    
    def mul_runs_BR(self, n):
        h_vec = create_h_vec(self.p1_history, self.p2_history, n)
        for i in range(n):
            run_BR(self.p1_strats, self.p2_strats, self.p1_params, self.p2_params, h_vec[i][0], h_vec[i][1], self.p1_payoffs, self.p2_payoffs, self.rounds, self.params_vec, self.init_params, actual_init=self.actual_init, σ_vec=self.σ_vec, random=self.random_params, lower_vec=self.lower_vec, upper_vec=self.upper_vec)
        return h_vec
    
    def run_LBR(self):
        run_LBR(self.p1_strats, self.p2_strats, self.p1_params, self.p2_params, self.p1_history, self.p2_history, self.p1_payoffs, self.p2_payoffs, self.rounds, self.params_vec, self.init_params, actual_init=self.actual_init, σ_vec=self.σ_vec, random=self.random_params, lower_vec=self.lower_vec, upper_vec=self.upper_vec)
        return [self.p1_history, self.p2_history]
    
    def mul_runs_LBR(self, n):
        h_vec = create_h_vec(self.p1_history, self.p2_history, n)
        for i in range(n):
            run_LBR(self.p1_strats, self.p2_strats, self.p1_params, self.p2_params, h_vec[i][0], h_vec[i][1], self.p1_payoffs, self.p2_payoffs, self.rounds, self.params_vec, self.init_params, actual_init=self.actual_init, σ_vec=self.σ_vec, random=self.random_params, lower_vec=self.lower_vec, upper_vec=self.upper_vec)
        return h_vec
    
    
    def run_LK(self):
        run_LK(self.p1_strats, self.p2_strats, self.p1_params, self.p2_params, self.p1_history, self.p2_history, self.p1_payoffs, self.p2_payoffs, self.rounds, self.params_vec, self.init_params, actual_init=self.actual_init, σ_vec=self.σ_vec, random=self.random_params, lower_vec=self.lower_vec, upper_vec=self.upper_vec)
        return [self.p1_history, self.p2_history]
    
    def run_LLK(self):
        run_LLK(self.p1_strats, self.p2_strats, self.p1_params, self.p2_params, self.p1_history, self.p2_history, self.p1_payoffs, self.p2_payoffs, self.rounds, self.params_vec, self.init_params, actual_init=self.actual_init, σ_vec=self.σ_vec, random=self.random_params, lower_vec=self.lower_vec, upper_vec=self.upper_vec)
        return [self.p1_history, self.p2_history]
    
    def mul_runs_LK(self, n):
        h_vec = create_h_vec(self.p1_history, self.p2_history, n)
        for i in range(n):
            run_LK(self.p1_strats, self.p2_strats, self.p1_params, self.p2_params, h_vec[i][0], h_vec[i][1], self.p1_payoffs, self.p2_payoffs, self.rounds, self.params_vec, self.init_params, actual_init=self.actual_init, σ_vec=self.σ_vec, random=self.random_params, lower_vec=self.lower_vec, upper_vec=self.upper_vec)
        return h_vec
    
    def mul_runs_LLK(self, n):
        h_vec = create_h_vec(self.p1_history, self.p2_history, n)
        for i in range(n):
            run_LLK(self.p1_strats, self.p2_strats, self.p1_params, self.p2_params, h_vec[i][0], h_vec[i][1], self.p1_payoffs, self.p2_payoffs, self.rounds, self.params_vec, self.init_params, actual_init=self.actual_init, σ_vec=self.σ_vec, random=self.random_params, lower_vec=self.lower_vec, upper_vec=self.upper_vec)
        return h_vec

    def run_PCHM(self):
        run_PCHM(self.p1_strats, self.p2_strats, self.p1_params, self.p2_params, self.p1_history, self.p2_history, self.p1_payoffs, self.p2_payoffs, self.rounds, self.params_vec, self.init_params, actual_init=self.actual_init, σ_vec=self.σ_vec, random=self.random_params, lower_vec=self.lower_vec, upper_vec=self.upper_vec)
        return [self.p1_history, self.p2_history]
    
    def mul_runs_PCHM(self, n):
        h_vec = create_h_vec(self.p1_history, self.p2_history, n)
        for i in range(n):
            run_PCHM(self.p1_strats, self.p2_strats, self.p1_params, self.p2_params, h_vec[i][0], h_vec[i][1], self.p1_payoffs, self.p2_payoffs, self.rounds, self.params_vec, self.init_params, actual_init=self.actual_init, σ_vec=self.σ_vec, random=self.random_params, lower_vec=self.lower_vec, upper_vec=self.upper_vec)
        return h_vec

    def run_JPCHM(self):
        run_JPCHM(self.p1_strats, self.p2_strats, self.p1_params, self.p2_params, self.p1_history, self.p2_history, self.p1_payoffs, self.p2_payoffs, self.rounds, self.params_vec, self.init_params, actual_init=self.actual_init, σ_vec=self.σ_vec, random=self.random_params, lower_vec=self.lower_vec, upper_vec=self.upper_vec)
        return [self.p1_history, self.p2_history]
    
    def mul_runs_JPCHM(self, n):
        h_vec = create_h_vec(self.p1_history, self.p2_history, n)
        for i in range(n):
            run_JPCHM(self.p1_strats, self.p2_strats, self.p1_params, self.p2_params, h_vec[i][0], h_vec[i][1], self.p1_payoffs, self.p2_payoffs, self.rounds, self.params_vec, self.init_params, actual_init=self.actual_init, σ_vec=self.σ_vec, random=self.random_params, lower_vec=self.lower_vec, upper_vec=self.upper_vec)
        return h_vec
    
    def run_LJPCHM(self):
        run_LJPCHM(self.p1_strats, self.p2_strats, self.p1_params, self.p2_params, self.p1_history, self.p2_history, self.p1_payoffs, self.p2_payoffs, self.rounds, self.params_vec, self.init_params, actual_init=self.actual_init, σ_vec=self.σ_vec, random=self.random_params, lower_vec=self.lower_vec, upper_vec=self.upper_vec)
        return [self.p1_history, self.p2_history]
    
    def mul_runs_LJPCHM(self, n):
        h_vec = create_h_vec(self.p1_history, self.p2_history, n)
        for i in range(n):
            run_LJPCHM(self.p1_strats, self.p2_strats, self.p1_params, self.p2_params, h_vec[i][0], h_vec[i][1], self.p1_payoffs, self.p2_payoffs, self.rounds, self.params_vec, self.init_params, actual_init=self.actual_init, σ_vec=self.σ_vec, random=self.random_params, lower_vec=self.lower_vec, upper_vec=self.upper_vec)
        return h_vec

    def run_L1PCHM(self):
        run_L1PCHM(self.p1_strats, self.p2_strats, self.p1_params, self.p2_params, self.p1_history, self.p2_history, self.p1_payoffs, self.p2_payoffs, self.rounds, self.params_vec, self.init_params, actual_init=self.actual_init, σ_vec=self.σ_vec, random=self.random_params, lower_vec=self.lower_vec, upper_vec=self.upper_vec)
        return [self.p1_history, self.p2_history]
    
    def mul_runs_L1PCHM(self, n):
        h_vec = create_h_vec(self.p1_history, self.p2_history, n)
        for i in range(n):
            run_L1PCHM(self.p1_strats, self.p2_strats, self.p1_params, self.p2_params, h_vec[i][0], h_vec[i][1], self.p1_payoffs, self.p2_payoffs, self.rounds, self.params_vec, self.init_params, actual_init=self.actual_init, σ_vec=self.σ_vec, random=self.random_params, lower_vec=self.lower_vec, upper_vec=self.upper_vec)
        return h_vec

    def run_LPCHM(self):
        run_LPCHM(self.p1_strats, self.p2_strats, self.p1_params, self.p2_params, self.p1_history, self.p2_history, self.p1_payoffs, self.p2_payoffs, self.rounds, self.params_vec, self.init_params, actual_init=self.actual_init, σ_vec=self.σ_vec, random=self.random_params, lower_vec=self.lower_vec, upper_vec=self.upper_vec)
        return [self.p1_history, self.p2_history]
    
    def mul_runs_LPCHM(self, n):
        h_vec = create_h_vec(self.p1_history, self.p2_history, n)
        for i in range(n):
            run_LPCHM(self.p1_strats, self.p2_strats, self.p1_params, self.p2_params, h_vec[i][0], h_vec[i][1], self.p1_payoffs, self.p2_payoffs, self.rounds, self.params_vec, self.init_params, actual_init=self.actual_init, σ_vec=self.σ_vec, random=self.random_params, lower_vec=self.lower_vec, upper_vec=self.upper_vec)
        return h_vec
    
    def run_EWA(self):
        run_EWA(self.p1_strats, self.p2_strats, self.p1_params, self.p2_params, self.p1_history, self.p2_history, self.p1_payoffs, self.p2_payoffs, self.p1_As, self.p2_As, self.p1_Ns, self.p2_Ns, self.rounds, self.params_vec, self.init_params, actual_init=self.actual_init, σ_vec=self.σ_vec, random=self.random_params, lower_vec=self.lower_vec, upper_vec=self.upper_vec)
        return [self.p1_history, self.p2_history]
    
    def mul_runs_EWA(self, n):
        h_vec = create_h_vec(self.p1_history, self.p2_history, n)
        for i in range(n):
            run_EWA(self.p1_strats, self.p2_strats, self.p1_params, self.p2_params, h_vec[i][0], h_vec[i][1], self.p1_payoffs, self.p2_payoffs, self.p1_As, self.p2_As, self.p1_Ns, self.p2_Ns, self.rounds, self.params_vec, self.init_params, actual_init=self.actual_init, σ_vec=self.σ_vec, random=self.random_params, lower_vec=self.lower_vec, upper_vec=self.upper_vec)
        return h_vec
    
    # def mul_runs_model(self, model, n):
    #     if model == "BR":
    #         return self.mul_runs_BR(n)
    #     elif model == "LK":
    #         return self.mul_runs_LK(n)
    #     elif model == "LPCHM":
    #         return self.mul_runs_LPCHM(n)
    #     elif model == "EWA":
    #         return self.mul_runs_EWA(n)
    #     else: 
    #         print("Invalid model too mul_runs_model") 

#%% Analyze funcs
# def flatten_h(hists):
#     flat_hist = np.zeros((len(hists), len(hists[0][0])*(len(hists[0][0][0]) + len(hists[0][1][0]))))
#     flatten_h_jit(hists,flat_hist)
#     return flat_hist

def sample_fun(dist):
    np.random.seed()
    if isinstance(dist, tuple):
        return np.random.uniform(low=dist[0], high=dist[1])
    else: 
        return np.random.choice(dist)
        
def flatten_h(hists):
    return np.array([np.append(np.array(hist[0]).ravel(), np.array(hist[1]).ravel()) for hist in hists])

def flatten_single_hist(hist):
    return np.append(np.array(hist[0]).ravel(), np.array(hist[1]).ravel())

def gen_kde(simulated, bw=0.07):
    flat = flatten_h(simulated)
    kde = KernelDensity(bandwidth=bw)
    kde.fit(flat)
    return kde 

def gen_kde_from_flat(simulated, bw=0.07):
    kde = KernelDensity(bandwidth=bw)
    kde.fit(simulated)
    return kde 

def calc_from_kde(kde, hist):
    flat_actual = np.array([flatten_single_hist(hist)])
    res = kde.score(flat_actual)
    if np.isnan(res):
        res = np.finfo('d').min 
    return res

def calc_avg_from_kde(kde, hists):
    tot_ll = 0
    for hist in hists:
        tot_ll += calc_from_kde(kde, hist)
    return(tot_ll/len(hists))

def id_generator(size=10, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))  

def test_model(actuals, model, payoffs, init_params, periods, n_runs, p1_size, p2_size, params, random_params=True, single_actual=True, mul_pop_types=False, init_strats=False):
    np.random.seed()
    res_vec = []
    if model == "BR":
        if random_params:
            σ_vec = np.array([params["p_sd"], params["ε_sd"]])
        else:
            σ_vec = np.array([])
        μ_vec = np.array([params["p"], params["ε"]])
        pop = Population(payoffs[0], payoffs[1], periods, p1_size, p2_size, params_vec=μ_vec, σ_vec=σ_vec, lower_vec=[0.,0.], upper_vec=[1.,1.], random_params=random_params, init_strats=init_strats) 
        hists = pop.mul_runs_BR(n_runs)
        
    elif model == "LK":
        if random_params:
            σ_vec = np.array([params["p_sd"], params["τ_sd"]])
        else:
            σ_vec = np.array([])
        μ_vec = np.array([params["p"], params["τ"]])
        pop = Population(payoffs[0], payoffs[1], periods, p1_size, p2_size, params_vec=μ_vec, σ_vec=σ_vec, lower_vec=[0.,0.], upper_vec=[1.,5.], random_params=random_params, init_strats=init_strats)
        hists = pop.mul_runs_LK(n_runs)
        
    elif model == "PCHM":
        if random_params:
            σ_vec = np.array([params["p_sd"], params["τ_sd"]])
        else:
            σ_vec = np.array([])
        μ_vec = np.array([params["p"], params["τ"]])
        pop = Population(payoffs[0], payoffs[1], periods, p1_size, p2_size, params_vec=μ_vec, σ_vec=σ_vec, lower_vec=[0.,0.], upper_vec=[1.,5.], random_params=random_params, init_strats=init_strats)
        hists = pop.mul_runs_PCHM(n_runs)
    # elif model == "LPCHM":
    #     if random_params:
    #         σ_vec = np.array([params["p_sd"], params["τ_sd"], params["λ_sd"]])
    #     else:
    #         σ_vec = np.array([])
    #     μ_vec = np.array([params["p"], params["τ"], params["λ"]])
    #     pop = Population(payoffs[0], payoffs[1], periods, p1_size, p2_size, params_vec=μ_vec, σ_vec=σ_vec, lower_vec=[0.,0.,0.], upper_vec=[1.,5.,20.], random_params=random_params, init_strats=init_strats)
    #     hists = pop.mul_runs_LPCHM(n_runs)
    elif model == "EWA":
        if random_params:
            σ_vec = np.array([params["p_sd"], params["λ_sd"], params["φ_sd"],params["ρ_sd"], params["δ_sd"]])
        else:
            σ_vec = np.array([])
        μ_vec = np.array([params["p"], params["λ"], params["φ"], params["ρ"], params["δ"]])
        pop = Population(payoffs[0], payoffs[1], periods, p1_size, p2_size, params_vec=μ_vec, σ_vec=σ_vec, lower_vec=[0.,0.,0.,0.,0.], upper_vec=[1.,20.,1.,1.,1.], random_params=random_params, init_strats=init_strats)
        hists = pop.mul_runs_EWA(n_runs)
        
    

    if single_actual:
        kde_001 = gen_kde(hists, bw=0.01)
        kde_002 = gen_kde(hists, bw=0.02)
        kde_003 = gen_kde(hists, bw=0.03)
        kde_005 = gen_kde(hists, bw=0.05)
        kde_007 = gen_kde(hists, bw=0.07)
        kde_01 = gen_kde(hists, bw=0.1)
        l001 = calc_from_kde(kde_001, actuals) 
        l002 = calc_from_kde(kde_002, actuals) 
        l003 = calc_from_kde(kde_003, actuals) 
        l005 = calc_from_kde(kde_005, actuals)
        l007 = calc_from_kde(kde_007, actuals)
        l01 = calc_from_kde(kde_01, actuals)
        res_vec.append({**params, "period":periods, "model":model, "actual_id":"single","l001":l001, "l002":l002, "l003":l003, "l005":l005, "l007":l007, "l01":l01, "n_sims":n_runs})
    
    if (not single_actual) and (not mul_pop_types):
        kde_001 = gen_kde(hists, bw=0.01)
        kde_002 = gen_kde(hists, bw=0.02)
        kde_003 = gen_kde(hists, bw=0.03)
        kde_005 = gen_kde(hists, bw=0.05)
        kde_007 = gen_kde(hists, bw=0.07)
        kde_01 = gen_kde(hists, bw=0.1)
        for i in range(len(actuals)):
            l001 = calc_from_kde(kde_001, actuals[i]) 
            l002 = calc_from_kde(kde_002, actuals[i]) 
            l003 = calc_from_kde(kde_003, actuals[i]) 
            l005 = calc_from_kde(kde_005, actuals[i])
            l007 = calc_from_kde(kde_007, actuals[i])
            l01 = calc_from_kde(kde_01, actuals[i])
            res_vec.append({**params, "period":periods, "model":model, "actual_id":i, "l001":l001, "l002":l002, "l003":l003, "l005":l005, "l007":l007, "l01":l01, "n_sims":n_runs})
    
    if (not single_actual) and mul_pop_types:
        kde_001 = gen_kde(hists, bw=0.01)
        kde_002 = gen_kde(hists, bw=0.02)
        kde_003 = gen_kde(hists, bw=0.03)
        kde_005 = gen_kde(hists, bw=0.05)
        kde_007 = gen_kde(hists, bw=0.07)
        kde_01 = gen_kde(hists, bw=0.1)
        for pop_type, actuals_vec in actuals.items():
            for i in range(len(actuals_vec)):
                l001 = calc_from_kde(kde_001, actuals_vec[i]) 
                l002 = calc_from_kde(kde_002, actuals_vec[i]) 
                l003 = calc_from_kde(kde_003, actuals_vec[i]) 
                l005 = calc_from_kde(kde_005, actuals_vec[i])
                l007 = calc_from_kde(kde_007, actuals_vec[i])
                l01 = calc_from_kde(kde_01, actuals_vec[i])
                res_vec.append({**params, "period":periods, "model":model, "pop_type":pop_type, "actual_id":i, "l001":l001, "l002":l002, "l003":l003, "l005":l005, "l007":l007, "l01":l01, "n_runs":n_runs})
    return res_vec

def test_model_n_times(n_sims, actuals, model, payoffs, init_params, periods, n_runs, p1_size, p2_size, param_space, ranfdom_params=True, single_actual=True, mul_pop_types=False, init_strats_dict=False):
    res_vec = []
    np.random.seed()
    start_time = time.time()
    for i in range(n_sims):
        params = dict()
        params["param_id"] = id_generator()
        for p_name, p_dist in param_space[model].items():
            params[p_name] = sample_fun(p_dist)
        for gid in payoffs:
            params["gid"] = gid
            if not init_strats_dict:
                params["init"] = False
                res_vec.extend(test_model(actuals[gid], model, payoffs[gid], init_params, periods, n_runs, p1_size, p2_size, params, random_params=random_params, single_actual=single_actual, mul_pop_types=mul_pop_types))
            else:
                params["init"] = True
                res_vec.extend(test_model(actuals[gid], model, payoffs[gid], init_params, periods, n_runs, p1_size, p2_size, params, random_params=random_params, single_actual=single_actual, mul_pop_types=mul_pop_types, init_strats=init_strats_dict[gid]))
    print("-- %d sims of %s which took %s seconds" %(n_sims, model, time.time() - start_time))
    return res_vec
    


def run_test(actual_plays, payoffs, param_spaces, n_runs, periods, p1_size, p2_size, random_params=True, single_actual=True, mul_pop_types=False, cores=mp.cpu_count(), n_per_cpu=1, init_params=[1.5, 0.5], init_strats_dict=False):
    print("Running test for %d runs %d sims per %d cpus" %(n_runs, n_per_cpu, cores))
    res_vec = []
    for model in param_spaces:
        start_time = time.time()
        pool = mp.Pool(processes=cores)
        res_mp = [pool.apply_async(test_model_n_times, args=(n_per_cpu, actual_plays, model, payoffs, init_params, periods, n_runs, p1_size, p2_size, param_spaces[model], random_params, single_actual, mul_pop_types, init_strats_dict)) for x in range(cores)]
        res_unflattened = [p.get() for p in res_mp]
        pool.close()
        pool.join()
        for res in res_unflattened:
            res_vec.extend(res)
        print("Done with %s which took %s seconds. In total %d sims of %d runs" %(model, time.time() - start_time, cores*n_per_cpu, n_runs))
    return res_vec
