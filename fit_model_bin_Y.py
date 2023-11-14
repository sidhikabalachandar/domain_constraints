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
    parser.add_argument('--num_warmup_iter', type=int)
    parser.add_argument('--num_sampling_iter', type=int)
    parser.add_argument('--num_chains', type=int, default=4)
    parser.add_argument('--job_id', type=int, default=0) # unique identifier for job
    parser.add_argument('--sigma', type=float) # set to -1 if you want sigma to float
    parser.add_argument('--alpha', type=float) # set to -1 if you want alpha to float
    parser.add_argument('--prevalence_constraint_weight', type=float, default=0) # set to 0 if you don't want prevalence
    parser.add_argument('--N', type=int, default=-1) # don't include if you want full dataset
    parser.add_argument('--file_path', type=str) # path to data
    parser.add_argument('--save_path', type=str) # path to folder where stan samples should be saved
    parser.add_argument('--model', type=str) # path to stan model file
    parser.add_argument('--Z_type', type=str, choices=['normal', 'uniform']) 
    parser.add_argument('--num_not_sparse', type=int)
    args = parser.parse_args()
    return args

def main():
    args = get_args()

    # get model
    stan_file = args.model
    with open(stan_file) as file:
        stan_code = file.read()

    # get data
    file = open(args.file_path, 'rb')
    simulated_data = pickle.load(file)
    file.close()
    
    print("job_id:", args.job_id)
    print("data file:", args.file_path)
    print("num_chains:", args.num_chains)
    print("num_warmup_iter:", args.num_warmup_iter)
    print("num_sampling_iter:", args.num_sampling_iter)

    # get observed data and parameters
    if args.N == -1:
        N = len(simulated_data['observed_data']['T'])
    else:
        N = args.N
    M = simulated_data['observed_data']['M']
    T = simulated_data['observed_data']['T'][:N]
    Y = simulated_data['observed_data']['Y'][:N]
    true_prevalence = simulated_data['observed_data']['Y'].mean()
    sigma = simulated_data['parameters']['sigma']
    alpha = simulated_data['parameters']['alpha']
    beta_Y = simulated_data['parameters']['beta_Y']
    beta_delta = simulated_data['parameters']['beta_delta']
    print("true prevalence", true_prevalence)
    print("p(T): %2.3f" % T.mean())
    print("p(Y): %2.3f" % Y.mean())
    print("p(Y=1|T=1): %2.3f" % (Y[T==1].mean()))
    print("p(Y=1|T=0): %2.3f" % (Y[T==0].mean()))

    # prevalence constraint
    simulated_data['observed_data']['prevalence_constraint_weight'] = args.prevalence_constraint_weight
    simulated_data['observed_data']['true_prevalence'] = true_prevalence
    if args.sigma != -1:
        simulated_data['observed_data']['known_sigma'] = sigma
    else:
        simulated_data['observed_data']['known_sigma'] = -1
    if args.alpha != -1:
        simulated_data['observed_data']['known_alpha'] = alpha
    else:
        simulated_data['observed_data']['known_alpha'] = -1

    # expertise constraint
    not_zeroed_out_beta_delta_indices = [i + 1 for i in range(args.num_not_sparse)]
    N_not_zeroed_out_beta_delta_indices = len(not_zeroed_out_beta_delta_indices)
    simulated_data['observed_data']['N_not_zeroed_out_beta_delta_indices'] = N_not_zeroed_out_beta_delta_indices
    simulated_data['observed_data']['not_zeroed_out_beta_delta_indices'] = not_zeroed_out_beta_delta_indices
            
    print()
    print("Model Properties:")
    print("true sigma:", sigma)
    print("true alpha:", alpha)
    print("model sigma:", simulated_data['observed_data']['known_sigma'])
    print("model alpha:", simulated_data['observed_data']['known_alpha'])
    print("prevalence constraint weight:", args.prevalence_constraint_weight)
    print("true prevalence:", true_prevalence)
    print("N_not_zeroed_out_beta_delta_indices:", N_not_zeroed_out_beta_delta_indices)
    print("not_zeroed_out_beta_delta_indices:", not_zeroed_out_beta_delta_indices)
    print('beta_Y:', beta_Y)
    print('beta_delta:', beta_delta)
    print('N:', N)
    
    data = {"N":N,
            "M":M, 
            "num_T1_Y1":np.sum((T == 1) & (Y == 1)),
            "num_T1_Y0":np.sum((T == 1) & (Y == 0)),
            "num_T0":np.sum(T == 0),
            "N_not_zeroed_out_beta_delta_indices": N_not_zeroed_out_beta_delta_indices,
            "not_zeroed_out_beta_delta_indices": not_zeroed_out_beta_delta_indices,
            "X":simulated_data['observed_data']['X'][:N], 
            "T":T.astype(int), 
            "Y":Y.astype(int),
            "T1_Y1_idxs":np.where((T == 1) & (Y == 1))[0] + 1,
            "T1_Y0_idxs":np.where((T == 1) & (Y == 0))[0] + 1,
            "T0_idxs":np.where(T == 0)[0] + 1, 
            'prevalence_constraint_weight':simulated_data['observed_data']['prevalence_constraint_weight'],
            'true_prevalence':simulated_data['observed_data']['true_prevalence'],
            'known_sigma': simulated_data['observed_data']['known_sigma'],
            'known_alpha': simulated_data['observed_data']['known_alpha']
           }

    # build and fit model
    model = stan.build(stan_code, data=data)
    fit = model.sample(num_chains=args.num_chains, num_warmup=args.num_warmup_iter, num_samples=args.num_sampling_iter)
    df = fit.to_frame()
    df.to_csv(os.path.join(args.save_path, "gen_data_{}_fit_{}.csv".format(args.Z_type, args.job_id)))

    # get arviz summary
    az_data = az.from_pystan(fit)
    az_df = az.summary(fit)
    az_df.to_csv(os.path.join(args.save_path, "az_summ_gen_data_{}_fit_{}.csv".format(args.Z_type, args.job_id)))
    if args.Z_type == 'uniform':
        summ = az.summary(fit, var_names = ['beta_Y', 'beta_delta', 'sigma', 'implied_prevalence'])
    elif args.Z_type == 'normal':
        summ = az.summary(fit, var_names = ['beta_Y', 'beta_delta', 'sigma_to_use', 'alpha_to_use', 'implied_prevalence'])
    summ = summ.drop(columns = summ.columns.difference(['mean','sd', 'r_hat']))
    print('summary:')
    print(summ)
    
    
if __name__ == "__main__":
    main()