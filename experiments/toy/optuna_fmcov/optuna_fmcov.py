import warnings

import numpy as np
import optuna
import pandas as pd
import torch
from rsvote.data.matrix import RollCallMatrix
from rsvote.models.fmcov import FMCov

# Ignore some sklearn convergence warnings for short trials if necessary
warnings.filterwarnings("ignore")

def objective(trial):
    """
    Optuna objective function to tune the FM-COV model.
    """
    n_legislators = 100
    n_votes = 200

    # 1. Generate a Synthetic Roll Call Matrix to simulate a legislature
    np.random.seed(42)  # For deterministic dataset across trials
    # Probability distribution: 40% Yea, 40% Nay, 20% Abstain
    votes = np.random.choice([0, 1, np.nan], size=(n_legislators, n_votes), p=[0.4, 0.4, 0.2])

    # 2. Add categorical covariates (as recommended by the FM-COV thesis)
    # Using the `cov_` prefix so `RollCallMatrix` detects them as embedding targets
    parties = ["Democrat", "Republican", "Independent"]
    states = ["SP", "RJ", "MG", "BA", "RS"]

    leg_df = pd.DataFrame({
        "id": range(n_legislators),
        "cov_party": np.random.choice(parties, size=n_legislators),
        "cov_state": np.random.choice(states, size=n_legislators)
    })

    rc_df = pd.DataFrame({
        "id": range(n_votes),
        "cov_topic": np.random.choice(["Economy", "Healthcare", "Defense"], size=n_votes)
    })

    # Build Train/Test configuration
    matrix = RollCallMatrix(votes, leg_df, rc_df)
    train_mat, val_mat = matrix.train_test_split(test_size=0.2, random_state=42)

    # 3. Specify Optuna Hyperparameters matching Silveira (2025) Appendix 3.4
    lr = trial.suggest_float("lr", 5e-4, 5e-2, log=True)
    l1_alpha = trial.suggest_float("l1_alpha", 1e-7, 1e-3, log=True)
    l1_p = trial.suggest_float("l1_p", 1e-7, 1e-3, log=True)
    l1_q = trial.suggest_float("l1_q", 1e-7, 1e-3, log=True)
    n_factors = trial.suggest_int("n_factors", 0, 3)

    model = FMCov(n_factors=n_factors)

    try:
        model.fit(
            X=train_mat,
            X_val=val_mat,
            epochs=50,  # Keeping epochs low purely to execute the hyperparameter sweep quickly
            lr=lr,
            lambda_alpha=l1_alpha,
            lambda_p=l1_p,
            lambda_q=l1_q,
            verbose=False # Rely on summary validation instead of printing every 10 epochs
        )
    except Exception as e:
        print(f"Trial failed: {e}")
        raise optuna.exceptions.TrialPruned()

    # 4. Extract out-of-sample Negative Log Likelihood (Binary Crossentropy)
    val_user_idx, val_item_idx, val_labels = val_mat.to_pytorch_tensors()
    val_user_idx = val_user_idx.to(model.device)
    val_item_idx = val_item_idx.to(model.device)
    val_labels = val_labels.to(model.device)

    # Reconstruct the Covariate Index matchers
    val_user_covs, _ = val_mat.get_user_covariates()
    val_item_covs, _ = val_mat.get_item_covariates()
    val_t_user_covs = {k: torch.tensor(v[val_user_idx.cpu().numpy()], dtype=torch.long, device=model.device) for k, v in val_user_covs.items()}
    val_t_item_covs = {k: torch.tensor(v[val_item_idx.cpu().numpy()], dtype=torch.long, device=model.device) for k, v in val_item_covs.items()}

    model.eval()
    with torch.no_grad():
        logits = model(val_user_idx, val_item_idx, val_t_user_covs, val_t_item_covs)
        loss = torch.nn.BCEWithLogitsLoss()(logits, val_labels)

    # Optuna will minimize this validation loss
    return loss.item()

if __name__ == "__main__":
    import os
    print("="*60)
    print("Executing Bayesian Optimization for FM-COV architecture")
    print("Testing hyperparameter regimes across L1 Regularizations and Embeddings...")
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
    print(f"Saved results to {out_path}")
