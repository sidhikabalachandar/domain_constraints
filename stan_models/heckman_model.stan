// code adapted from: https://www.jchau.org/2021/02/07/fitting-the-heckman-selection-model-with-stan-and-r/
// Z_noise, u ~ N([0, 0], [[sigma^2, rho],[rho, 1]])
// Y ~ X * beta_Y + Z_noise
// T ~ 1[Z * beta_T + u > 0]
data {
  // dimensions
  int<lower=1> N;
  int<lower=1, upper=N> N_y;
  int<lower=1> M;
  // covariates
  matrix[N_y, M] X;
  matrix[N, M] Z;
  // responses
  int<lower=0, upper=1> T[N];
  vector[N_y] Y;
  int<lower=1> N_zeroed_out_beta_delta_indices;
  int<lower=-1, upper=M> zeroed_out_beta_delta_indices[N_zeroed_out_beta_delta_indices]; // pass in [-1] if you don't want any zeroed out indices. 
  real true_prevalence; 
  real<lower=0> prevalence_constraint_weight; 
}
parameters {
  vector[M] beta_Y;
  vector[M] beta_T;
  real<lower=0> rho;
  real<lower=0> sigma;
}
transformed parameters{
  real implied_prevalence = mean(Z * beta_Y);
}
model {
  // naive (truncated) priors
  beta_Y ~ normal(0, 3);
  beta_T ~ normal(0, 3);
  rho ~ normal(0, 3);
  sigma ~ normal(0, 3);
  {
    // expertise constraint
    vector[M] masked_beta_Y = beta_Y;
    for(i in 1:N_zeroed_out_beta_delta_indices) {
      int idx = zeroed_out_beta_delta_indices[i];
      if (idx > 0) {
        masked_beta_Y[idx] = beta_T[idx] * (sigma^2) / rho;
      }
    }
    
    // log-likelihood
    vector[N_y] XbetaY = X * masked_beta_Y;
    vector[N] ZbetaT = Z * beta_T;
    int ny = 1;
    for(n in 1:N) {
      if(T[n] > 0) {
        target += normal_lpdf(Y[ny] | XbetaY[ny], sigma) + log(Phi((ZbetaT[n] + rho / (sigma^2) * (Y[ny] - XbetaY[ny])) / sqrt(1 - (rho^2/ (sigma^2)))));
        ny += 1; 
      }
      else {
        target += log(Phi(-ZbetaT[n]));
      }
    }
  }
  
  // prevalence constraint
  if(prevalence_constraint_weight > 0){
    target += -prevalence_constraint_weight*square(implied_prevalence - true_prevalence);
  }
}