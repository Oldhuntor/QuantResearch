import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.statespace.mlemodel import MLEModel

class TimeVaryingBetaModel(MLEModel):
    def __init__(self, endog, exog):
        super().__init__(endog, exog)
        self.k_exog = exog.shape[1]
        self.state_names = [f'beta_{i}' for i in range(self.k_exog)]
        self.initialize_state()

    def state_space_representation(self):
        state_intercept = np.zeros((self.k_exog,))
        transition = np.eye(self.k_exog)  # Random walk for beta_t
        selection = np.eye(self.k_exog)
        state_cov = np.eye(self.k_exog) * 0.01  # Process noise
        return state_intercept, transition, selection, state_cov

# Generate example data
np.random.seed(42)
n = 100
X = np.random.randn(n, 2)
beta_true = np.array([0.5, -0.3])
Y = X @ beta_true + 0.1 * np.random.randn(n)

# Fit the model
model = TimeVaryingBetaModel(Y, X)
results = model.fit()
beta_estimates = results.filtered_state.T
