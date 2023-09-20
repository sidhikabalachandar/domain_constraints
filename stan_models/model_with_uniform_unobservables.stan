// this model lets sigma float but fixes alpha = 1. 
// Z ~ Uniform(0, sigma)
// r = X * beta_Y + Z
// Y ~ Bernoulli(sigmoid(r))
// T ~ Bernoulli(sigmoid(r + X * beta_delta))
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
  real<lower=0> true_prevalence; 
  real<lower=0> prevalence_constraint_weight; 
}
parameters {
  vector[M] beta_Y;
  vector[N_not_zeroed_out_beta_delta_indices] beta_delta;
  real<lower=0> sigma;
}
transformed parameters{
    real implied_prevalence = mean((sigma - log(1 + exp(-X * beta_Y)) + log(exp(-X * beta_Y - sigma) + 1))/sigma);
}
model {
    // normal priors
    beta_Y ~ normal(0, 3);
    beta_delta ~ normal(0, 3);
    sigma ~ normal(0, 3);

    vector[N] XbY = X * beta_Y; 
    vector[N] XbT = XbY + X[:,not_zeroed_out_beta_delta_indices] * beta_delta; // model is T ~ sigmoid(XbT + Z) = sigmoid(X*beta_Y + X*beta_delta + Z) = sigmoid(r + X * beta_delta)

    // p(T=1, Y=1|X)
    target += sum(log((sigma*(exp(XbT[T1_Y1_idxs]) - exp(XbY[T1_Y1_idxs])) - exp(XbT[T1_Y1_idxs]).*log((exp(XbY[T1_Y1_idxs]) + 1).*exp(-XbT[T1_Y1_idxs])) + exp(XbT[T1_Y1_idxs]).*log((exp(XbY[T1_Y1_idxs] + sigma) + 1).*exp(-XbT[T1_Y1_idxs] - sigma)) + exp(XbY[T1_Y1_idxs]).*log((exp(XbT[T1_Y1_idxs]) + 1).*exp(-XbT[T1_Y1_idxs])) - exp(XbY[T1_Y1_idxs]).*log((exp(XbT[T1_Y1_idxs] + sigma) + 1).*exp(-XbT[T1_Y1_idxs] - sigma)))./(sigma*(exp(XbT[T1_Y1_idxs]) - exp(XbY[T1_Y1_idxs])))));
    // p(T=1, Y=0|X)
    target += sum(log((-log((exp(XbT[T1_Y0_idxs]) + 1).*exp(-XbT[T1_Y0_idxs])) + log((exp(XbY[T1_Y0_idxs]) + 1).*exp(-XbT[T1_Y0_idxs])) + log((exp(XbT[T1_Y0_idxs] + sigma) + 1).*exp(-XbT[T1_Y0_idxs] - sigma)) - log((exp(XbY[T1_Y0_idxs] + sigma) + 1).*exp(-XbT[T1_Y0_idxs] - sigma))).*exp(XbT[T1_Y0_idxs])./(sigma*(exp(XbT[T1_Y0_idxs]) - exp(XbY[T1_Y0_idxs])))));
    // p(T=0|X)
    target += sum(log((log(1 + exp(-XbT[T0_idxs])) - log(exp(-XbT[T0_idxs] - sigma) + 1))/sigma));

    // prevalence constraint
    if(prevalence_constraint_weight > 0){
      target += -prevalence_constraint_weight*square(implied_prevalence - true_prevalence);
    }
}