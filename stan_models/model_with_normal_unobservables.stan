data {
  int<lower=1> N;         // number of people
  int<lower=1> M;         // number of features
  int<lower=0> num_T1_Y1;         // number tested and diagnosed
  int<lower=0> num_T1_Y0;         // number tested and undiagnosed
  int<lower=0> num_T0;         // number untested
  int<lower=1> N_zeroed_out_beta_delta_indices;
  int<lower=-1, upper=M> zeroed_out_beta_delta_indices[N_zeroed_out_beta_delta_indices]; // pass in [-1] if you don't want any zeroed out indices. 
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
  vector[M] beta_delta;
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
    vector[M] masked_beta_delta = beta_delta;
    for(j in 1:N_zeroed_out_beta_delta_indices){
      if(zeroed_out_beta_delta_indices[j] != -1){
        masked_beta_delta[zeroed_out_beta_delta_indices[j]] *= 0; // remember index 1 is intercept
      }
    }
    
    real T1_Y1_ll; 
    real T1_Y0_ll;
    real T0_ll;
    {
        vector[N] risk_score = X * beta_Y + Z; 
        vector[N] p_Y = inv_logit(risk_score);
        vector[N] p_T = inv_logit(alpha_to_use * risk_score + X * masked_beta_delta);
        T1_Y1_ll = sum(log(inv_logit(risk_score)[T1_Y1_idxs] .* p_T[T1_Y1_idxs]));
        T1_Y0_ll = sum(log((1 - p_Y[T1_Y0_idxs]) .* p_T[T1_Y0_idxs]));
        T0_ll = sum(log(1 - p_T[T0_idxs]));
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

    target += T1_Y1_ll;
    target += T1_Y0_ll;
    target += T0_ll;

    // prevalence constraint
    if(prevalence_constraint_weight > 0){
      target += prevalence_constraint;
    }
}