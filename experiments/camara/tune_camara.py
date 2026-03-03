import os
import pandas as pd
import numpy as np
import torch
import optuna
import warnings
warnings.filterwarnings("ignore")

from rsvote.data.matrix import RollCallMatrix
from rsvote.models import FMCov

def objective_factory(matrix):
    def objective(trial):
        lr = trial.suggest_float("lr", 1e-3, 1e-1, log=True)
        l1_alpha = trial.suggest_float("l1_alpha", 1e-7, 1e-2, log=True)
        l1_p = trial.suggest_float("l1_p", 1e-7, 1e-2, log=True)
        l1_q = trial.suggest_float("l1_q", 1e-7, 1e-2, log=True)
        n_factors = trial.suggest_int("n_factors", 0, 20)

        cv_losses = []
        for seed in [42, 123, 999]:
            train_mat, val_mat = matrix.train_test_split(test_size=0.1, random_state=seed)
            model = FMCov(n_factors=n_factors, epochs=50, lr=lr, lambda_alpha=l1_alpha, lambda_p=l1_p, lambda_q=l1_q, verbose=False)
            try:
                model.fit(X=train_mat, X_val=val_mat)
            except Exception:
                raise optuna.exceptions.TrialPruned()
    
            probs = model.predict_proba(val_mat)
            _, _, labels = val_mat.to_pytorch_tensors()
            loss = torch.nn.functional.binary_cross_entropy(probs, labels)
            cv_losses.append(loss.item())
            
        return np.mean(cv_losses)
    return objective

def tune_camara():
    print("\n" + "="*60)
    print("Loading Camara D03 (2023 subset for temporal speed)...")
    data_path = os.path.join(os.path.dirname(__file__), "data/D03_Camara_Deputados_PLEN.csv")
    if not os.path.exists(data_path):
        print(f"Data not found at {data_path}. Skipping.")
        return

    df = pd.read_csv(data_path)
    df['Data'] = pd.to_datetime(df['Data'])
    df = df[df['Data'].dt.year >= 2023]
    
    df['voto_bin'] = df['voto'].map({'Sim': 1.0, 'Não': 0.0})
    df = df.dropna(subset=['voto_bin'])
    
    wide = df.pivot_table(index='deputado_nome', columns='idVotacao', values='voto_bin')
    
    legis_df = df[['deputado_nome', 'deputado_siglaPartido', 'deputado_siglaUf']].drop_duplicates(subset='deputado_nome')
    legis_df = legis_df.set_index('deputado_nome').reindex(wide.index).reset_index()
    legis_df = legis_df.rename(columns={'deputado_siglaPartido': 'cov_partido', 'deputado_siglaUf': 'cov_uf'})
    
    matrix = RollCallMatrix(wide.values, legislators=legis_df)
    
    study = optuna.create_study(direction="minimize")
    study.optimize(objective_factory(matrix), n_trials=10)
    
    out_str = f"Best Camara Params: {study.best_params}"
    print("\n" + out_str)
    with open(os.path.join(os.path.dirname(__file__), "camara_results.txt"), "w") as f:
        f.write(out_str + "\n")

def tune_adin():
    print("\n" + "="*60)
    print("Loading ADIn (Brazilian Supreme Court)...")
    data_path = os.path.join(os.path.dirname(__file__), "data/D01. ADINs - Jeferson Mariano Jurisdicao_constitucional_no_Brasil_1966.csv")
    if not os.path.exists(data_path):
        print(f"Data not found at {data_path}. Skipping.")
        return

    df = pd.read_csv(data_path)
    wide = df.pivot_table(index='id_votante', columns='id_votacao', values='voto')
    
    matrix = RollCallMatrix(wide.values)
    
    study = optuna.create_study(direction="minimize")
    study.optimize(objective_factory(matrix), n_trials=10)
    
    out_str = f"Best ADIn (Supreme Court) Params: {study.best_params}"
    print("\n" + out_str)
    with open(os.path.join(os.path.dirname(__file__), "adin_results.txt"), "w") as f:
        f.write(out_str + "\n")

if __name__ == "__main__":
    tune_camara()
    tune_adin()
