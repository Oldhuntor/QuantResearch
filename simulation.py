from model.composite_GPs2 import ThreeComponentGP

import torch
import matplotlib.pyplot as plt

def run_simulation(model,
                   X: torch.Tensor,
                   Y: torch.Tensor,
                   true_error:torch.Tensor,
                   window_size: int = 50,
                   change_points: torch.Tensor = None,
                   num_samples: int = 1000):
    """
    Simulate an online streaming scenario:
      - Use an initial window of size 'window_size' to train the model.
      - For each subsequent data point t in [window_size, len(X)):
          1) Predict one step ahead for X[t], Y[t].
          2) Then do 'online_update' with the newly arrived data (X[t], Y[t]).
      - Collect the predictions and compute an overall MSE.

    Args:
        model: An instance of ThreeComponentGP or similar that supports online_update().
        X, Y: Full dataset (1D tensors).
        window_size: Number of points to use in the initial window.
        num_samples: Number of posterior samples for GP predictions (for credible intervals).

    Returns:
        predictions: a list of one-step-ahead predicted means
        actuals:     the actual Y[t] values used for comparison
        ci_lowers:   list of lower bound for 95% CI
        ci_uppers:   list of upper bound for 95% CI
    """
    X = X.double().clone()
    Y = Y.double().clone()
    N = len(X)

    # 1) Initialize the model on the first 'window_size' points
    # (We assume 'model' is already constructed with the first window in __init__,
    #  or you can do something like:)
    # model = ThreeComponentGP(X[:window_size], Y[:window_size])
    # model.train_model(num_epochs=100, learning_rate=0.01)
    # But let's suppose you've already done an initial training externally.

    predictions = []
    ci_lowers = []
    ci_uppers = []
    actuals = []

    # 2) Stream the data from index=window_size onward
    for t in range(window_size, N):
        # One-step-ahead prediction at X[t], Y[t]
        x_eval = X[t].unsqueeze(0)  # shape (1,)
        y_eval = Y[t].unsqueeze(0)  # shape (1,)

        # Use the model's predict method
        pred_dict = model.predict(x_eval, y_eval, num_samples=num_samples)

        final_pred = pred_dict['final']
        mean_pred_t = final_pred['mean']  # shape (1,)
        ci_t = final_pred['credible_intervals']  # shape (2, 1)

        predictions.append(mean_pred_t.item())
        ci_lowers.append(ci_t[0, 0].item())
        ci_uppers.append(ci_t[1, 0].item())
        # actuals.append(Y[t].item())

        # 3) Online update with the newly arrived data
        model.online_update(X[t].item(), Y[t].item())

    # Compute MSE
    # predictions_torch = torch.tensor(predictions, dtype=torch.float64)
    # actuals_torch = torch.tensor(actuals, dtype=torch.float64)
    # mse = torch.mean((predictions_torch - actuals_torch) ** 2).item()
    # print(f"One-step-ahead MSE: {mse:.4f}")

    # 4) Plot predictions vs actual
    steps = range(window_size, N)
    plt.figure(figsize=(15, 5))
    plt.plot(steps, true_error, 'k-', label='Actual')
    plt.plot(steps, predictions, 'b--', label='Predicted Mean')
    plt.fill_between(steps, ci_lowers, ci_uppers, color='blue', alpha=0.2, label='95% CI')
    # plt.title(f"One-step-ahead GP Predictions (MSE={mse:.4f})")
    # Draw vertical lines at change_points
    if change_points is not None:
        # If change_points are indices, we can just do plt.axvline(idx).
        # If they represent some other x-values, we might map them to the step index.
        for cp in change_points:
            if 0 <= cp < N:
                plt.axvline(cp, color='red', linestyle='--', alpha=0.7, label='Change Point')
        # Trick: If we have multiple change_points, the label might appear multiple times.
        # We can handle that by labeling only once or customizing the legend manually.

    plt.xlabel("Time Index")
    plt.ylabel("Y")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return predictions, actuals, ci_lowers, ci_uppers


def run_simulation_retraining(
    X: torch.Tensor,
    Y: torch.Tensor,
    window_size: int = 50,
    num_samples: int = 1000,
    model_class=None,
    verbose: bool = False
):
    """
    Simulate an online streaming scenario by re-initializing and fully retraining
    the ThreeComponentGP (or a similar GP class) on each sliding window.
    Then do one-step-ahead prediction on the next point.

    Args:
        X, Y: full dataset (1D tensors).
        window_size: number of points used in each training window.
        num_samples: for posterior sampling (confidence intervals).
        model_class: the GP class (e.g., ThreeComponentGP) to instantiate each time.
        verbose: if True, prints training progress at each step.

    Returns:
        predictions: a list of one-step-ahead predicted means
        actuals:     the actual Y[t] values
        ci_lowers:   list of 95% CI lower bounds
        ci_uppers:   list of 95% CI upper bounds
    """
    X = X.double().clone()
    Y = Y.double().clone()
    N = len(X)

    predictions = []
    ci_lowers   = []
    ci_uppers   = []
    actuals     = []

    # We'll do a rolling approach from t = window_size to t = N-1
    # For each t:
    #   1) Re-init a new model with [t-window_size : t] as training data
    #   2) train_model()
    #   3) predict one-step-ahead at X[t], Y[t]
    for t in range(window_size, N):
        # Training window
        X_window = X[t - window_size : t]
        Y_window = Y[t - window_size : t]

        # Build a fresh model instance for this window
        # We assume your model_class signature is model_class(X, Y, window_size=window_size)
        model = model_class(X_window, Y_window, window_size=window_size)

        # Train the model from scratch
        model.train_model(num_epochs=50, learning_rate=0.01, verbose=verbose)  # or your desired hyperparams

        # Now do one-step-ahead prediction at X[t], Y[t]
        x_eval = X[t].unsqueeze(0)  # shape (1,)
        y_eval = Y[t].unsqueeze(0)
        pred_dict = model.predict(x_eval, y_eval, num_samples=num_samples)

        final_pred = pred_dict['final']
        mean_pred_t = final_pred['mean']  # shape (1,)
        ci_t = final_pred['credible_intervals']  # shape (2,1)

        predictions.append(mean_pred_t.item())
        ci_lowers.append(ci_t[0, 0].item())
        ci_uppers.append(ci_t[1, 0].item())
        actuals.append(Y[t].item())

    # Compute MSE
    predictions_torch = torch.tensor(predictions, dtype=torch.float64)
    actuals_torch = torch.tensor(actuals, dtype=torch.float64)
    mse = torch.mean((predictions_torch - actuals_torch)**2).item()
    print(f"One-step-ahead MSE (full retraining): {mse:.4f}")

    # Plot
    steps = range(window_size, N)
    plt.figure(figsize=(15,5))
    plt.plot(steps, actuals, 'k-', label='Actual')
    plt.plot(steps, predictions, 'b--', label='Predicted Mean')
    plt.fill_between(steps, ci_lowers, ci_uppers, color='blue', alpha=0.2, label='95% CI')

    if change_points is not None:
        # If change_points are indices, we can just do plt.axvline(idx).
        # If they represent some other x-values, we might map them to the step index.
        for cp in change_points:
            if 0 <= cp < N:
                plt.axvline(cp, color='red', linestyle='--', alpha=0.7, label='Change Point')

    plt.title(f"One-step-ahead GP Predictions (Full Retraining), MSE={mse:.4f}")
    plt.xlabel("Index / Time")
    plt.ylabel("Y")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return predictions, actuals, ci_lowers, ci_uppers


def run_simulation_hybrid(
    X: torch.Tensor,
    Y: torch.Tensor,
    window_size: int,
    model_class,
    true_error: torch.Tensor,
    retrain_every: int = 10,
    num_epochs: int = 50,
    learning_rate: float = 0.01,
    num_samples: int = 1000,
    change_points = None,
    verbose: bool = False
):
    """
    Hybrid approach:
      - Use a sliding window of size 'window_size'.
      - For t in [window_size, len(X)):
          1) Predict one-step-ahead at X[t], Y[t].
          2) If (t - window_size) % retrain_every == 0: full re-init and train from scratch
             else: do a quick online_update(X[t], Y[t]) or partial adaptation
      - Return predictions, actuals, CI bounds, and plot final results.

    Args:
        X, Y: full dataset (1D tensors)
        window_size: initial window size
        model_class: the GP class, e.g. ThreeComponentGP
        retrain_every: every 'retrain_every' steps, we do a full re-training from scratch
        num_epochs: full training epochs when re-init
        learning_rate: for full training
        num_samples: for posterior sampling
        verbose: print training progress or not

    Returns:
        predictions, actuals, ci_lowers, ci_uppers
        :param change_points:
    """
    X = X.double().clone()
    Y = Y.double().clone()

    N = len(X)

    predictions = []
    ci_lowers   = []
    ci_uppers   = []
    actuals     = []

    # Step 1) Initialize model on the first window
    model = model_class(X[:window_size], Y[:window_size], window_size=window_size)
    model.train_model(num_epochs=num_epochs, learning_rate=learning_rate, verbose=verbose)

    for t in range(window_size, N):
        # 2) Predict one-step-ahead
        x_eval = X[t].unsqueeze(0)
        y_eval = Y[t].unsqueeze(0)
        pred_dict = model.predict(x_eval, y_eval, num_samples=num_samples)

        final_pred = pred_dict['error']
        mean_pred_t = final_pred['mean']  # shape (1,)
        ci_t = final_pred['credible_intervals']  # shape (2,1)

        predictions.append(mean_pred_t.item())
        ci_lowers.append(ci_t[0, 0].item())
        ci_uppers.append(ci_t[1, 0].item())
        # actuals.append(Y[t].item())
        actuals.append(true_error[t].item())


        # 3) Sliding Window Update Strategy
        step_idx = t - window_size  # how many steps we've moved beyond the initial window
        if step_idx % retrain_every == 0 and step_idx > 0:
            # **Full re-training** from scratch on [t-window_size : t]
            X_window = X[t-window_size:t]
            Y_window = Y[t-window_size:t]
            model = model_class(X_window, Y_window, window_size=window_size)
            model.train_model(num_epochs=num_epochs, learning_rate=learning_rate, verbose=verbose)
        else:
            # **Online Update**: Just do a quick sliding-window approach
            # model.online_update(X[t].item(), Y[t].item())
            pass

    # Compute MSE
    predictions_torch = torch.tensor(predictions, dtype=torch.float64)
    actuals_torch = torch.tensor(actuals, dtype=torch.float64)
    mse = torch.mean((predictions_torch - actuals_torch)**2).item()
    print(f"Hybrid One-step-ahead MSE: {mse:.4f}")

    # Plot
    steps = range(window_size, N)
    plt.figure(figsize=(10,6))
    plt.plot(steps, actuals, 'k-', label='Actual')
    plt.plot(steps, predictions, 'b--', label='Predicted Mean')
    plt.fill_between(steps, ci_lowers, ci_uppers, color='blue', alpha=0.2, label='95% CI')
    if change_points is not None:
        # If change_points are indices, we can just do plt.axvline(idx).
        # If they represent some other x-values, we might map them to the step index.
        for cp in change_points:
            if 0 <= cp < N:
                plt.axvline(cp, color='red', linestyle='--', alpha=0.7, label='Change Point')
    plt.title(f"Hybrid GP Predictions: Full Retrain Every {retrain_every} Steps (MSE={mse:.4f})")
    plt.xlabel("Index / Time")
    plt.ylabel("error")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return predictions, actuals, ci_lowers, ci_uppers



if __name__ == "__main__":
    data_points = 1000
    window_size = 100
    change_points = (260,564, 777)
    time, X_coin, Y_coin, beta_true, mu_true = ThreeComponentGP.generate_cointegration_data(
        T=data_points,
        seed=1,
        noise_std=0.5,
        change_points=change_points,
        beta_values=(1.5, 3, 2.5, 1),
        mu_values=(9, 8, 7, 6),
        alpha=1
    )
    true_error = Y_coin - beta_true*X_coin - mu_true
    X_train = X_coin[:window_size]
    Y_train = Y_coin[:window_size]
    model = ThreeComponentGP(X_train, Y_train, window_size=window_size)
    model.train_model(num_epochs=50, learning_rate=0.01)

    # Now simulate streaming for the rest
    # preds, actuals, ci_low, ci_high = run_simulation(
    #     model,
    #     X_coin,
    #     Y_coin,
    #     change_points=change_points,
    #     true_error=true_error,
    #     window_size=window_size,
    #     num_samples=500
    # )
    #
    # predictions, actuals, ci_low, ci_high = run_simulation_retraining(
    #     X_coin,
    #     Y_coin,
    #     window_size=window_size,
    #     num_samples=500,
    #     model_class=ThreeComponentGP,
    #     verbose=False
    # )
    #
    # plot_cointegration_data(time, X_coin, Y_coin, beta_true, mu_true)
    run_simulation_hybrid(
        X_coin,
        Y_coin,
        window_size=window_size,
        model_class=ThreeComponentGP,  # your GP class
        retrain_every=100,
        true_error=true_error,
        change_points=change_points,
        num_epochs=100,
        learning_rate=0.01,
        num_samples=500,
        verbose=True
    )