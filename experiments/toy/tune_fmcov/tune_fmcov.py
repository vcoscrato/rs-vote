import warnings
import os

import numpy as np
import optuna
import pandas as pd
import torch
from rsvote.data.matrix import RollCallMatrix
from rsvote.models import FMCov

warnings.filterwarnings("ignore")

def objective(trial):
    """
    Optuna objective function to tune the FM-COV model.
    """
    n_legislators = 100
    n_votes = 200

    rng = np.random.default_rng(42)  # For deterministic dataset across trials
    votes = rng.choice([0, 1, np.nan], size=(n_legislators, n_votes), p=[0.4, 0.4, 0.2])

    parties = ["Democrat", "Republican", "Independent"]
    states = ["SP", "RJ", "MG", "BA", "RS"]

    leg_df = pd.DataFrame({
        "id": range(n_legislators),
        "cov_party": rng.choice(parties, size=n_legislators),
        "cov_state": rng.choice(states, size=n_legislators)
    })

    rc_df = pd.DataFrame({
        "id": range(n_votes),
        "cov_topic": rng.choice(["Economy", "Healthcare", "Defense"], size=n_votes)
    })

    matrix = RollCallMatrix(votes, leg_df, rc_df)
    train_mat, val_mat = matrix.train_test_split(test_size=0.2, random_state=42)

    lr = trial.suggest_float("lr", 5e-4, 5e-2, log=True)
    l1_alpha = trial.suggest_float("l1_alpha", 1e-7, 1e-3, log=True)
    l1_p = trial.suggest_float("l1_p", 1e-7, 1e-3, log=True)
    l1_q = trial.suggest_float("l1_q", 1e-7, 1e-3, log=True)
    n_factors = trial.suggest_int("n_factors", 0, 3)

    model = FMCov(
        n_factors=n_factors,
        epochs=50,
        lr=lr,
        lambda_alpha=l1_alpha,
        lambda_p=l1_p,
        lambda_q=l1_q,
        verbose=False
    )

    try:
        model.fit(X=train_mat, X_val=val_mat)
    except Exception as e:
        print(f"Trial failed: {e}")
        raise optuna.exceptions.TrialPruned()

    # Use model's internal validation results if available, or compute simply
    # The fit method already prints validation metrics, but Optuna needs the value returned.
    # We can use predict_proba to get out-of-sample log-loss
    probs = model.predict_proba(val_mat)
    _, _, labels = val_mat.to_pytorch_tensors()
    
    # Calculate Binary Cross Entropy manually for the return
    loss = torch.nn.functional.binary_cross_entropy(probs, labels)

    return loss.item()

if __name__ == "__main__":
    print("="*60)
    print("Executing Bayesian Optimization for FM-COV architecture")
    print("="*60)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=10)

    out_lines = ["="*60, "Best hyperparameters found by Optuna:"]
    for key, value in study.best_params.items():
        out_lines.append(f"  {key}: {value}")
    out_lines.append("="*60)
    
    out_text = "\n".join(out_lines)
    print("\n" + out_text)
    
    out_path = os.path.join(os.path.dirname(__file__), "optuna_results.txt")
    with open(out_path, "w") as f:
        f.write(out_text)
