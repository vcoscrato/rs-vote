import os
import pandas as pd
import numpy as np
import torch
import optuna
import warnings
warnings.filterwarnings("ignore")

from rsvote.data.matrix import RollCallMatrix
from rsvote.models import FMCov

def map_sen90_votes(v):
    if pd.isna(v): return np.nan
    v = float(v)
    if v in [1, 2, 3]: return 1.0
    if v in [4, 5, 6]: return 0.0
    return np.nan

def tune_sen90():
    print("="*60)
    print("Loading sen90 (90th US Senate)...")
    base_path = os.path.dirname(__file__)
    votes_path = os.path.join(base_path, "data/sen90_votes.csv")
    legis_path = os.path.join(base_path, "data/sen90_legis.csv")
    
    if not os.path.exists(votes_path) or not os.path.exists(legis_path):
        print("Data files not found. Skipping.")
        return

    votes_df = pd.read_csv(votes_path)
    legis_df = pd.read_csv(legis_path)
    
    legis_df = legis_df.rename(columns={"party": "cov_party"})
    votes = votes_df.map(map_sen90_votes).values
    
    matrix = RollCallMatrix(votes, legislators=legis_df)
    
    def objective(trial):
        lr = trial.suggest_float("lr", 1e-3, 1e-1, log=True)
        l1_alpha = trial.suggest_float("l1_alpha", 1e-7, 1e-2, log=True)
        l1_p = trial.suggest_float("l1_p", 1e-7, 1e-2, log=True)
        l1_q = trial.suggest_float("l1_q", 1e-7, 1e-2, log=True)
        n_factors = trial.suggest_int("n_factors", 0, 3)

        cv_losses = []
        for seed in [42, 123, 999]:
            train_mat, val_mat = matrix.train_test_split(test_size=0.1, random_state=seed)
            model = FMCov(n_factors=n_factors, epochs=75, lr=lr, lambda_alpha=l1_alpha, lambda_p=l1_p, lambda_q=l1_q, verbose=False)
            try:
                model.fit(X=train_mat, X_val=val_mat)
            except Exception:
                raise optuna.exceptions.TrialPruned()
    
            probs = model.predict_proba(val_mat)
            _, _, labels = val_mat.to_pytorch_tensors()
            loss = torch.nn.functional.binary_cross_entropy(probs, labels)
            cv_losses.append(loss.item())
            
        return np.mean(cv_losses)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=10)
    
    out_str = f"Best SEN90 Params: {study.best_params}"
    print("\n" + out_str)
    with open(os.path.join(os.path.dirname(__file__), "sen90_results.txt"), "w") as f:
        f.write(out_str + "\n")


if __name__ == "__main__":
    tune_sen90()
