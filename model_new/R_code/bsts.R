# Bayesian Structural Time Series in R with Stan: Detailed Comparison

library(rstan)
library(ggplot2)

# Set random seed
set.seed(0)

# Generate synthetic data
T <- 100
x <- runif(T) * 10
beta_true <- sin(seq_len(T) / 10) + 2
plot(beta_true)
mu_true <- cos(seq_len(T) / 20) * 5
epsilon <- rnorm(T, 0, 2)
y <- beta_true * x + mu_true + epsilon

# Stan model
stan_model <- "
data {
    int<lower=0> T;       // number of time steps
    vector[T] x;          // input variable
    vector[T] y;          // observed output
}

parameters {
    real<lower=0> sigma_eps;    // noise standard deviation
    real<lower=0> sigma_beta;   // beta random walk std dev
    real<lower=0> sigma_mu;     // mu random walk std dev
    vector[T] beta;             // time-varying beta
    vector[T] mu;               // time-varying mu
}

model {
    // Priors
    sigma_eps ~ cauchy(0, 5);
    sigma_beta ~ cauchy(0, 1);
    sigma_mu ~ cauchy(0, 5);

    // Random walks for beta and mu
    beta[1] ~ normal(0, 10);
    mu[1] ~ normal(0, 10);
    for (t in 2:T) {
        beta[t] ~ normal(beta[t-1], sigma_beta);
        mu[t] ~ normal(mu[t-1], sigma_mu);
    }

    // Likelihood
    for (t in 1:T) {
        y[t] ~ normal(beta[t] * x[t] + mu[t], sigma_eps);
    }
}

generated quantities {
    vector[T] y_pred;
    for (t in 1:T) {
        y_pred[t] = normal_rng(beta[t] * x[t] + mu[t], sigma_eps);
    }
}"

# Prepare data for Stan
stan_data <- list(T = T, x = x, y = y)

# Run Stan model
fit <- stan(model_code = stan_model,
            data = stan_data,
            iter = 2000,
            chains = 4)

# Extract posterior samples
beta_samples <- extract(fit)$beta
mu_samples <- extract(fit)$mu

# Calculate estimates and credible intervals
beta_mean <- apply(beta_samples, 2, mean)
beta_lower <- apply(beta_samples, 2, function(x) quantile(x, 0.025))
beta_upper <- apply(beta_samples, 2, function(x) quantile(x, 0.975))

mu_mean <- apply(mu_samples, 2, mean)
mu_lower <- apply(mu_samples, 2, function(x) quantile(x, 0.025))
mu_upper <- apply(mu_samples, 2, function(x) quantile(x, 0.975))

# Compute performance metrics
# Root Mean Square Error (RMSE)
beta_rmse <- sqrt(mean((beta_true - beta_mean)^2))
mu_rmse <- sqrt(mean((mu_true - mu_mean)^2))

# Mean Absolute Error (MAE)
beta_mae <- mean(abs(beta_true - beta_mean))
mu_mae <- mean(abs(mu_true - mu_mean))

# Plotting
par(mfrow=c(2,2))

# Beta plots
plot(seq_len(T), beta_true, type = "l", col = "blue",
     xlab = "Time", ylab = "Beta", main = "True vs Estimated Beta")
lines(seq_len(T), beta_mean, col = "red")
lines(seq_len(T), beta_lower, col = "gray", lty = 2)
lines(seq_len(T), beta_upper, col = "gray", lty = 2)
#legend("topright", legend = c("True Beta", "Estimated Beta", "95% CI"),
#       col = c("blue", "red", "gray"), lty = c(1,1,2))

# Beta scatter plot
plot(beta_true, beta_mean,
     xlab = "True Beta", ylab = "Estimated Beta",
     main = "Beta: True vs Estimated")
abline(a = 0, b = 1, col = "red", lty = 2)

# Mu plots
plot(seq_len(T), mu_true, type = "l", col = "blue",
     xlab = "Time", ylab = "Mu", main = "True vs Estimated Mu")
lines(seq_len(T), mu_mean, col = "red")
lines(seq_len(T), mu_lower, col = "gray", lty = 2)
lines(seq_len(T), mu_upper, col = "gray", lty = 2)
#legend("topright", legend = c("True Mu", "Estimated Mu", "95% CI"),
#       col = c("blue", "red", "gray"), lty = c(1,1,2))

# Mu scatter plot
plot(mu_true, mu_mean,
     xlab = "True Mu", ylab = "Estimated Mu",
     main = "Mu: True vs Estimated")
abline(a = 0, b = 1, col = "red", lty = 2)

# Print performance metrics
cat("Beta Performance Metrics:\n")
cat("RMSE:", beta_rmse, "\n")
cat("MAE:", beta_mae, "\n\n")

cat("Mu Performance Metrics:\n")
cat("RMSE:", mu_rmse, "\n")
cat("MAE:", mu_mae, "\n")