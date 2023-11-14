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
    parser.add_argument('--prevalence_constraint_weight', type=float, default=0) # set to 0 if you don't want prevalence
    parser.add_argument('--N', type=int, default=-1) # don't incclude if you want full dataset
    parser.add_argument('--file_path', type=str) # path to data
    parser.add_argument('--save_path', type=str) # path to folder where stan samples should be saved
    parser.add_argument('--model', type=str) # path to stan model file
    parser.add_argument('--num_sparse', type=int)
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
    X = simulated_data['observed_data']['X'][:N]
    T = simulated_data['observed_data']['T'][:N]
    Y = simulated_data['observed_data']['Y'][:N]
    true_prevalence = simulated_data['observed_data']['Y'].mean()
    sigma = simulated_data['parameters']['sigma']
    rho = simulated_data['parameters']['rho']
    beta_Y = simulated_data['parameters']['beta_Y']
    beta_T = simulated_data['parameters']['beta_T']
    print("true prevalence", true_prevalence)
    print("p(T): %2.3f" % T.mean())
    print("p(Y): %2.3f" % Y.mean())
    print("p(Y=1|T=1): %2.3f" % (Y[T==1].mean()))
    print("p(Y=1|T=0): %2.3f" % (Y[T==0].mean()))

    # constraints
    simulated_data['observed_data']['prevalence_constraint_weight'] = args.prevalence_constraint_weight
    simulated_data['observed_data']['true_prevalence'] = true_prevalence
    if args.num_sparse > 0:
        zeroed_out_beta_delta_indices = [M - i for i in range(args.num_sparse)]
        zeroed_out_beta_delta_indices.reverse()
        N_zeroed_out_beta_delta_indices = len(zeroed_out_beta_delta_indices)
        simulated_data['observed_data']['N_zeroed_out_beta_delta_indices'] = N_zeroed_out_beta_delta_indices
        simulated_data['observed_data']['zeroed_out_beta_delta_indices'] = zeroed_out_beta_delta_indices
    else:
        zeroed_out_beta_delta_indices = [-1]
        N_zeroed_out_beta_delta_indices = len(zeroed_out_beta_delta_indices)
        simulated_data['observed_data']['N_zeroed_out_beta_delta_indices'] = N_zeroed_out_beta_delta_indices
        simulated_data['observed_data']['zeroed_out_beta_delta_indices'] = zeroed_out_beta_delta_indices
            
    print()
    print("Model Properties:")
    print("true sigma:", sigma)
    print("true rho:", rho)
    print("prevalence constraint weight:", args.prevalence_constraint_weight)
    print("true prevalence:", true_prevalence)
    print("N_zeroed_out_beta_delta_indices:", N_zeroed_out_beta_delta_indices)
    print("zeroed_out_beta_delta_indices:", zeroed_out_beta_delta_indices)
    print('beta_Y:', beta_Y)
    print('beta_T:', beta_T)
    print('N:', N)
    
    data = {"N":N,
            "N_y": T.sum(),
            "M":simulated_data['observed_data']['M'], 
            "X":X[T == 1], 
            "Z":X, 
            "T":T, 
            "Y":Y[T == 1],
            "N_zeroed_out_beta_delta_indices": N_zeroed_out_beta_delta_indices,
            "zeroed_out_beta_delta_indices": zeroed_out_beta_delta_indices,
            'prevalence_constraint_weight':simulated_data['observed_data']['prevalence_constraint_weight'],
            'true_prevalence':simulated_data['observed_data']['true_prevalence']
           }

    # build and fit model
    model = stan.build(stan_code, data=data)
    fit = model.sample(num_chains=args.num_chains, num_warmup=args.num_warmup_iter, num_samples=args.num_sampling_iter)
    df = fit.to_frame()
    df.to_csv(os.path.join(args.save_path, "gen_data_heckman_fit_{}.csv".format(args.job_id)))

    # get arviz summary
    az_data = az.from_pystan(fit)
    az_df = az.summary(fit)
    az_df.to_csv(os.path.join(args.save_path, "az_summ_gen_data_heckman_fit_{}.csv".format(args.job_id)))
    summ = az.summary(fit)
    summ = summ.drop(columns = summ.columns.difference(['mean','sd', 'r_hat']))
    print('summary:')
    print(summ)
    
    
if __name__ == "__main__":
    main()