import numpy as np
import stan
import arviz as az
import pandas as pd
import argparse
import matplotlib.pyplot as plt
import pickle
import os
from scipy.special import expit

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_datasets_to_save', type=int, default=200)
    parser.add_argument('--N', type=int, default=5000)
    parser.add_argument('--M', type=int, default=5)
    parser.add_argument('--rho_mean', type=float, default=1) 
    parser.add_argument('--sigma_mean', type=float, default=2) 
    parser.add_argument('--betaY_intercept_mean', type=float, default=-2) 
    parser.add_argument('--betaT_intercept_mean', type=float, default=2) 
    parser.add_argument('--intercept_std', type=float, default=0.5)
    parser.add_argument('--sigma_std', type=float, default=0.5)
    parser.add_argument('--rho_std', type=float, default=0.5) 
    parser.add_argument('--beta_std', type=float, default=1)
    parser.add_argument('--save_path', type=str) # path to where data should be saved
    parser.add_argument('--num_not_sparse', type=int, default=2)
    args = parser.parse_args()
    return args

def generate_simulated_data(N, M, sigma, rho, beta_std, betaY_intercept=None, betaT_intercept=None, betaDelta_0_except_for_these_idxs=None): 
    """
    N: number of observations
    M: number of features
    sigma: standard deviation of unobservables Z
    alpha: coefficient on Z in T. 
    Generative model: 

    X ~ N(0, 1) (plus an intercept column of ones)
    Z, u ~ N([0, 0], [[sigma^2, rho],[rho, 1]])
    Y ~ X * beta_Y + Z
    T ~ 1[X * beta_T + u > 0]
    """
    X = np.random.normal(size=(N, M - 1))
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    X = np.concatenate((np.ones((N, 1)), X), axis=1)
    beta_Y = np.random.normal(scale=beta_std, size=(M))
    if betaDelta_0_except_for_these_idxs is not None:
        assert 0 in betaDelta_0_except_for_these_idxs
        beta_T = np.random.normal(scale=beta_std, size=(M))
        for i in range(M):
            if i not in betaDelta_0_except_for_these_idxs:
                beta_T[i] = beta_Y[i] * rho / (sigma ** 2)
    else:
        beta_T = np.random.normal(scale=beta_std, size=(M))
    if betaY_intercept is not None:
        beta_Y[0] = betaY_intercept
    if betaT_intercept is not None:
        beta_T[0] = betaT_intercept
    Sigma = np.array([[sigma**2, rho], [rho, 1]])
    e = np.random.multivariate_normal(np.array([0,0]), Sigma, size=(N)) 
    Z = e[:,0]
    u = e[:,1] 
    T = (X @ beta_T + u > 0) + 0
    Y = (X @ beta_Y + Z)

    assert(Y.shape == (N,))
    assert(T.shape == (N,))
    
    return {'observed_data':{'X':X, 'Y':Y.flatten(), 'T':T.flatten(), 'N':N, 'M':M},
            'latent_data':{'Z':Z}, 
            'parameters':{'beta_Y':beta_Y, 'beta_T':beta_T, 'rho':rho, 'sigma':sigma, 'implied_prevalence':np.mean(Y)}}

def main():
    args = get_args()
    
    betaDelta_0_except_for_these_idxs = [i for i in range(args.num_not_sparse)]

    for i in range(args.num_datasets_to_save):
        rho = np.random.normal(loc=args.rho_mean, scale=args.rho_std)
        sigma = np.random.normal(loc=args.sigma_mean, scale=args.sigma_std)
        betaY_intercept = np.random.normal(loc=args.betaY_intercept_mean, scale=args.intercept_std)
        betaT_intercept = np.random.normal(loc=args.betaT_intercept_mean, scale=args.intercept_std)
        simulated_data = generate_simulated_data(N=args.N, 
                                                         M=args.M, 
                                                         sigma=sigma, 
                                                         rho=rho,
                                                         betaY_intercept=betaY_intercept, 
                                                         betaT_intercept=betaT_intercept,
                                                         betaDelta_0_except_for_these_idxs=betaDelta_0_except_for_these_idxs,
                                                         beta_std=args.beta_std)

        file = open('{}/heckman_N_{}_M_{}_y0_{}_t0_{}_s_{}_r_{}_istd_{}_sstd_{}_rstd_{}_bstd_{}_v{}.pkl'.format(args.save_path, args.N, args.M, args.betaY_intercept_mean, args.betaT_intercept_mean, args.sigma_mean, args.rho_mean, args.intercept_std, args.sigma_std, args.rho_std, args.beta_std, i), 'wb')
        pickle.dump(simulated_data, file)
        file.close()
    
    
if __name__ == "__main__":
    main()