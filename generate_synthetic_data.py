import numpy as np
import stan
import arviz as az
import pandas as pd
import argparse
import matplotlib.pyplot as plt
import pickle
import os	

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_datasets_to_save', type=int)
    parser.add_argument('--N', type=int)
    parser.add_argument('--M', type=int)
    parser.add_argument('--alpha_mean', type=float) 
    parser.add_argument('--sigma_mean', type=float) 
    parser.add_argument('--betaY_intercept_mean', type=float) 
    parser.add_argument('--betaDelta_intercept_mean', type=float) 
    parser.add_argument('--intercept_std', type=float)
    parser.add_argument('--sigma_std', type=float)
    parser.add_argument('--alpha_std', type=float) 
    parser.add_argument('--save_path', type=str) # path to where data should be saved
    parser.add_argument('--Z_type', type=str) 
    args = parser.parse_args()
    return args

def generate_simulated_data(N, M, sigma, alpha, Z_type, betaY_intercept=None, betaDelta_intercept=None, betaDelta_0_except_for_these_idxs=None): 
    """
    N: number of observations
    M: number of features
    sigma: standard deviation of unobservables Z
    alpha: coefficient on Z in T. 
    Generative model: 

    X ~ N(0, 1) (plus an intercept column of ones)
    Z ~ f(.|sigma)
    r = X * betaY + Z
    Y ~ Bernoulli(sigmoid(r))
    T ~ Bernoulli(sigmoid(alpha * r + X * betaDelta))
    """
    X = np.random.normal(size=(N, M - 1))
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    X = np.concatenate((np.ones((N, 1)), X), axis=1)
    if Z_type == 'normal':
        Z = np.random.normal(loc=0, scale=sigma, size=(N, 1))
    elif Z_type == 'uniform':
        Z = np.random.uniform(low=0, high=sigma, size=(N, 1))
    betaY = np.random.normal(size=(M, 1))
    
    # create betaDelta
    if betaDelta_0_except_for_these_idxs is not None:
        assert 0 in betaDelta_0_except_for_these_idxs
        betaDelta = np.zeros((M, 1))
        betaDelta[betaDelta_0_except_for_these_idxs, 0] = np.random.normal(size=(len(betaDelta_0_except_for_these_idxs)))
    else:
        betaDelta = np.random.normal(size=(M, 1))
        
    if betaY_intercept is not None:
        betaY[0] = betaY_intercept
    if betaDelta_intercept is not None:
        betaDelta[0] = betaDelta_intercept
        
    r = X @ betaY + Z
    p_Y = expit(r)
    Y = (np.random.random(p_Y.shape) < p_Y).astype(int)
    p_T = expit(alpha * r + X @ betaDelta)
    T = (np.random.random(p_T.shape) < p_T).astype(int)

    # scatterplot of p_T vs p_Y with title of mean T and mean Y
    if Y[T==1].mean() - Y.mean() > cutoff:
        plt.scatter(p_T, p_Y, alpha=0.1)
        plt.title('Mean T: ' + str(np.mean(T)) + ', Mean Y: ' + str(np.mean(Y)) + '\np(Y=1|T=1): ' + str(np.mean(Y[T == 1])) + ', p(Y=1|T=0): ' + str(np.mean(Y[T == 0])))
        plt.xlabel('p_T')
        plt.ylabel('p_Y')
        plt.show()
    
    return {'observed_data':{'X':X, 'Y':Y.flatten(), 'T':T.flatten(), 'N':N, 'M':M,
            'num_T1_Y1':np.sum((T == 1) & (Y == 1)), 'num_T1_Y0':np.sum((T == 1) & (Y == 0)), 
            'num_T0':np.sum(T == 0),
            'T1_Y1_idxs':np.where((T == 1) & (Y == 1))[0] + 1, 
            'T1_Y0_idxs':np.where((T == 1) & (Y == 0))[0] + 1,
            'T0_idxs':np.where(T == 0)[0] + 1},
            'latent_data':{'Z':Z, 'p_Y':p_Y, 'p_T':p_T}, 
            'parameters':{'betaY':betaY, 'betaDelta':betaDelta, 'alpha':alpha, 'sigma':sigma, 'implied_prevalence':np.mean(Y)}}

def main():
    args = get_args()
    
    betaDelta_0_except_for_these_idxs = [0, 1]

    for i in range(args.num_datasets_to_save):
        alpha = np.random.normal(loc=args.alpha_mean, scale=args.alpha_std)
        sigma = np.random.normal(loc=args.sigma_mean, scale=args.sigma_std)
        betaY_intercept = np.random.normal(loc=args.betaY_intercept_mean, scale=args.intercept_std)
        betaDelta_intercept = np.random.normal(loc=args.betaDelta_intercept_mean, scale=args.intercept_std)
        simulated_data = generate_simulated_data(N=args.N, 
                                                         M=args.M, 
                                                         sigma=sigma, 
                                                         alpha=alpha,
                                                         betaY_intercept=betaY_intercept, 
                                                         betaDelta_intercept=betaDelta_intercept,
                                                         betaDelta_0_except_for_these_idxs=betaDelta_0_except_for_these_idxs, 
                                                         Z_type=args.Z_type)

        file = open('{}/{}_unobservables_N_{}_y0_{}_d0_{}_istd_{}_sstd_{}_brstd_{}_v{}.pkl'.format(args.save_path, args.Z_type, cutoff, N, Y_intercept_mean, delta_intercept_mean, intercept_std, sigma_std, beta_risk_std, i), 'wb')
        pickle.dump(simulated_data, file)
        file.close()
    
    
if __name__ == "__main__":
    main()