import os
import pandas as pd
import numpy as np
import torch
import optuna
import warnings
warnings.filterwarnings("ignore")

from rsvote.data.matrix import RollCallMatrix
from rsvote.models.fmcov import FMCov

def map_sen90_votes(v):
    if pd.isna(v): return np.nan
    v = float(v)
    if v in [1, 2, 3]: return 1.0
    if v in [4, 5, 6]: return 0.0
    return np.nan

def optuna_sen90():
    print("="*60)
    print("Loading sen90 (90th US Senate)...")
    votes_df = pd.read_csv(os.path.join(os.path.dirname(__file__), "data/sen90_votes.csv"))
    legis_df = pd.read_csv(os.path.join(os.path.dirname(__file__), "data/sen90_legis.csv"))
    
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
            model = FMCov(n_factors=n_factors)
            try:
                model.fit(
                    X=train_mat, X_val=val_mat, epochs=75,
                    lr=lr, lambda_alpha=l1_alpha, lambda_p=l1_p, lambda_q=l1_q, verbose=False
                )
            except Exception:
                raise optuna.exceptions.TrialPruned()
    
            val_user_idx, val_item_idx, val_labels = val_mat.to_pytorch_tensors()
            val_user_idx = val_user_idx.to(model.device)
            val_item_idx = val_item_idx.to(model.device)
            val_labels = val_labels.to(model.device)
    
            val_user_covs, _ = val_mat.get_user_covariates()
            val_item_covs, _ = val_mat.get_item_covariates()
            val_t_user_covs = {k: torch.tensor(v[val_user_idx.cpu().numpy()], dtype=torch.long, device=model.device) for k, v in val_user_covs.items()}
            val_t_item_covs = {k: torch.tensor(v[val_item_idx.cpu().numpy()], dtype=torch.long, device=model.device) for k, v in val_item_covs.items()}
    
            model.eval()
            with torch.no_grad():
                logits = model(val_user_idx, val_item_idx, val_t_user_covs, val_t_item_covs)
                val_loss = torch.nn.BCEWithLogitsLoss()(logits, val_labels)
            cv_losses.append(val_loss.item())
            
        return np.mean(cv_losses)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=25)
    out_str = f"Best SEN90 Params: {study.best_params}"
    print("\n" + out_str)
    with open(os.path.join(os.path.dirname(__file__), "sen90_results.txt"), "w") as f:
        f.write(out_str + "\n")


if __name__ == "__main__":
    optuna_sen90()
