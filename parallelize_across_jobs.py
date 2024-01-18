import os
import argparse 

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--slurm_submission_script', type=str)
    parser.add_argument('--python_file', type=str)
    parser.add_argument('--num_warmup_iter', type=int)
    parser.add_argument('--num_sampling_iter', type=int)
    parser.add_argument('--num_chains', type=int, default=4)
    parser.add_argument('--job_id', type=int, default=0) # unique identifier for job
    parser.add_argument('--sigma', type=float) # set to -1 if you want sigma to float
    parser.add_argument('--alpha', type=float) # set to -1 if you want alpha to float
    parser.add_argument('--N', type=int, default=-1) # don't include if you want full dataset
    parser.add_argument('--M', type=int, default=5)
    parser.add_argument('--file_dir', type=str) # path to data
    parser.add_argument('--save_path', type=str) # path to folder where stan samples should be saved
    parser.add_argument('--model', type=str) # path to stan model file
    parser.add_argument('--Z_type', type=str, choices=['normal', 'uniform']) 
    parser.add_argument('--num_not_sparse', type=int)
    parser.add_argument('--starting_job_id', type=int, default=0)
    parser.add_argument('--num_jobs_to_sumbit', type=int, default=200)
    parser.add_argument('--prevalence_constraint_weight', type=float, default=0) # set to 0 if you don't want prevalence
    args = parser.parse_args()
    return args

def main():
    args = get_args()
    
    cmd = "sbatch --requeue {} python {} --num_warmup_iter {} \
                                         --num_sampling_iter {} \
                                         --num_chains {} \
                                         --N {} \
                                         --sigma {} \
                                         --alpha {} \
                                         --prevalence_constraint_weight {} \
                                         --num_not_sparse {} \
                                         --model {} \
                                         --file_path {} \
                                         --save_path {} \
                                         --Z_type {} \
                                         --job_id {}"
    
    cur_job_id = args.starting_job_id
    for i in range(args.num_jobs_to_sumbit):
        # change file name for different data
        file_path = os.path.join(args.file_dir, "uniform_unobservables_N_50_y0_-2.0_d0_2.0_istd_0.1_sstd_0.1_brstd_0.0_v{}.pkl".format(i)) 
        
        # submit unconstrained job
        formatted_cmd = cmd.format(args.slurm_submission_script,
                                   args.python_file,
                                   args.num_warmup_iter,
                                   args.num_sampling_iter,
                                   args.num_chains,
                                   args.N,
                                   args.sigma,
                                   args.alpha,
                                   0,
                                   args.M,
                                   args.model,
                                   file_path,
                                   args.save_path,
                                   args.Z_type,
                                   cur_job_id)          
        cur_job_id += 1
        os.system(formatted_cmd)
                                 
        # submit prevalence constrained job
        formatted_cmd = cmd.format(args.slurm_submission_script,
                                   args.python_file,
                                   args.num_warmup_iter,
                                   args.num_sampling_iter,
                                   args.num_chains,
                                   args.N,
                                   args.sigma,
                                   args.alpha,
                                   args.prevalence_constraint_weight,
                                   args.M,
                                   args.model,
                                   file_path,
                                   args.save_path,
                                   args.Z_type,
                                   cur_job_id)        
        cur_job_id += 1
        os.system(formatted_cmd)
                                 
        # submit expertise constrained job
        formatted_cmd = cmd.format(args.slurm_submission_script,
                                   args.python_file,
                                   args.num_warmup_iter,
                                   args.num_sampling_iter,
                                   args.num_chains,
                                   args.N,
                                   args.sigma,
                                   args.alpha,
                                   0,
                                   args.num_not_sparse,
                                   args.model,
                                   file_path,
                                   args.save_path,
                                   args.Z_type,
                                   cur_job_id)        
        cur_job_id += 1
        os.system(formatted_cmd)
    
if __name__ == "__main__":
    main()