// Z ~ Normal(0, sigma)
// r = X * beta_Y + Z
// Y ~ Bernoulli(sigmoid(r))
// T ~ Bernoulli(sigmoid(alpha * r + X * beta_delta))
data {
  int<lower=1> N;         // number of people
  int<lower=1> M;         // number of features
  int<lower=0> num_T1_Y1;         // number tested and diagnosed
  int<lower=0> num_T1_Y0;         // number tested and undiagnosed
  int<lower=0> num_T0;         // number untested
  int<lower=1> N_not_zeroed_out_beta_delta_indices;
  int<lower=-1, upper=M> not_zeroed_out_beta_delta_indices[N_not_zeroed_out_beta_delta_indices]; // must contain intercept (index 1) 
  matrix[N, M] X;   // predictor matrix
  int<lower=0,upper=1> T[N];   // tested indicators
  int<lower=0,upper=1> Y[N];   // diagnosed indicators
  int<lower=1,upper=N> T1_Y1_idxs[num_T1_Y1];    // indices tested and diagnosed
  int<lower=1,upper=N> T1_Y0_idxs[num_T1_Y0];    // indices tested and undiagnosed
  int<lower=1,upper=N> T0_idxs[num_T0];          // indices untested
  real<lower=0> prevalence_constraint_weight; 
  real<lower=0,upper=1> true_prevalence;
  real known_sigma; // if < 0, infer sigma
  real known_alpha; // if < 0, infer alpha. 
}
parameters {
  vector[M] beta_Y;
  vector[N_not_zeroed_out_beta_delta_indices] beta_delta;
  real<lower=0> inferred_alpha;
  real<lower=0> inferred_sigma; 
  vector[N] Z;
}
transformed parameters{
    real implied_prevalence = mean(inv_logit(X * beta_Y + Z));
    // options: pass in known_sigma and known_alpha. If either is > 0, we use that. Otherwise, we infer it. 
    real sigma_to_use; 
    real alpha_to_use; 
    if(known_sigma > 0){
      sigma_to_use = known_sigma; 
    }else{
      sigma_to_use = inferred_sigma;
    }
    if(known_alpha > 0){
      alpha_to_use = known_alpha;
    }else{
      alpha_to_use = inferred_alpha; 
    }
    
    real prevalence_constraint = -prevalence_constraint_weight*square(implied_prevalence - true_prevalence);
}
model {
    // normal priors on betas
    beta_Y ~ normal(0, 3);
    beta_delta ~ normal(0, 3);
    inferred_sigma ~ normal(0, 3); // this only gets used if we don't pass in known_sigma=-1.
    inferred_alpha ~ normal(0, 3); // this only gets used if we don't pass in known_alpha=-1. 
    Z ~ normal(0, sigma_to_use);

    vector[N] risk_score = X * beta_Y + Z; 
    vector[N] p_Y = inv_logit(risk_score);
    vector[N] p_T = inv_logit(alpha_to_use * risk_score + X[:,not_zeroed_out_beta_delta_indices] * beta_delta);

    target += sum(log(p_Y[T1_Y1_idxs] .* p_T[T1_Y1_idxs]));
    target += sum(log((1 - p_Y[T1_Y0_idxs]) .* p_T[T1_Y0_idxs]));
    target += sum(log(1 - p_T[T0_idxs]));

    // prevalence constraint
    if(prevalence_constraint_weight > 0){
      target += prevalence_constraint;
    }
}