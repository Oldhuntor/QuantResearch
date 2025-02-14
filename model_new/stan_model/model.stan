data {
  int<lower=1> T;           // Number of time steps
  vector[T] Y;              // Observations
  vector[T] X;              // Predictor
}

parameters {
  vector[T] beta;           // Dynamic regression coefficients
  vector[T] mu;             // Time-varying intercept
  real<lower=0> sigma;      // Observation noise
  real<lower=0> sigma_beta; // Std dev of beta's random walk
  real<lower=0> sigma_mu;   // Std dev of mu's random walk
}

model {
  // Priors
  beta[1] ~ normal(0, 10);
  mu[1] ~ normal(0, 10);
  sigma ~ cauchy(0, 1);
  sigma_beta ~ cauchy(0, 1);
  sigma_mu ~ cauchy(0, 1);

  // Dynamic evolution
  for (t in 2:T) {
    beta[t] ~ normal(beta[t-1], sigma_beta);
    mu[t] ~ normal(mu[t-1], sigma_mu);
  }

  // Observation model
  for (t in 1:T) {
    Y[t] ~ normal(beta[t] * X[t] + mu[t], sigma);
  }
}
