using Pkg
Pkg.add(["Turing", "DynamicHMC", "Distributions", "DataFrames", "Plots"])

using Turing, DynamicHMC, Distributions, DataFrames, Plots, Random

# Generate synthetic data (without autocorrelation)
Random.seed!(0)
T = 1:100
x = rand(100) * 10
beta_true = sin.(T ./ 10) .+ 2
mu_true = cos.(T ./ 20) .+ 5
epsilon = randn(100) .* 2
y = beta_true .* x .+ mu_true .+ epsilon

# Create DataFrame
df = DataFrame(y = y, x = x)

# Define Turing model
@model function time_varying_model(y, x)
    N = length(y)
    beta = zeros(N)
    mu = zeros(N)
    sigma ~ Exponential(1)

    beta[1] ~ Normal(0, 5)
    mu[1] ~ Normal(0, 5)

    for t in 2:N
        beta[t] ~ Normal(beta[t-1], 0.5)
        mu[t] ~ Normal(mu[t-1], 1)
    end

    for t in 1:N
        y[t] ~ Normal(beta[t] * x[t] + mu[t], sigma)
    end
end

# Create model instance
model = time_varying_model(df.y, df.x)

# Sample using DynamicHMC
chain = sample(model, DynamicHMC(1000), MCMCThreads()) # MCMCThreads for parallel sampling

# Extract posterior samples
beta_samples = group(chain, :beta).beta
mu_samples = group(chain, :mu).mu
sigma_samples = group(chain, :sigma).sigma

# Calculate posterior means
beta_mean = mean(beta_samples, dims=1)[:]
mu_mean = mean(mu_samples, dims=1)[:]
sigma_mean = mean(sigma_samples)

# Plotting
p1 = plot(df.y, label="y")
p2 = plot(T, beta_true, label="True Beta")
plot!(T, beta_mean, label="Estimated Beta")
p3 = plot(T, mu_true, label="True Mu")
plot!(T, mu_mean, label="Estimated Mu")

plot(p1, p2, p3, layout=(3,1), size=(800,600))

#Residual plot
residuals = df.y .- (beta_mean .* df.x .+ mu_mean)
plot(T, residuals)
plot!(title = "Residuals")