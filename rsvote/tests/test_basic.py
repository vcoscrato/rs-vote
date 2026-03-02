import pytest
import numpy as np
import pandas as pd
import torch

from rsvote.data import RollCallMatrix
from rsvote.models import WNominate, LogisticMatrixFactorization

def test_rollcallmatrix_initialization():
    # Test valid initialization
    votes = np.array([
        [1, 0, np.nan],
        [0, 1, 1]
    ])
    matrix = RollCallMatrix(votes)
    assert matrix.shape == (2, 3)
    
    # Test tensor conversion
    u_idx, i_idx, labels = matrix.to_pytorch_tensors()
    assert len(u_idx) == 5 # 5 non-NaN entries
    assert torch.all(labels == torch.tensor([1, 0, 0, 1, 1], dtype=torch.float32))

def test_rollcallmatrix_validation():
    # Test invalid values
    votes = np.array([[1, 2], [0, 1]]) # 2 is invalid
    with pytest.raises(ValueError):
        RollCallMatrix(votes)
        
def test_model_fitting():
    # Very basic synthetic data to ensure fit runs without error
    np.random.seed(42)
    votes = np.random.choice([0, 1, np.nan], size=(10, 15), p=[0.4, 0.4, 0.2])
    matrix = RollCallMatrix(votes)
    
    # Test W-NOMINATE
    model_wn = WNominate(n_dims=1)
    model_wn.fit(matrix, epochs=10) # minimal epochs for fast test
    assert model_wn._is_fitted
    assert model_wn.ideal_points_.shape == (10, 1)
    
    # Test LMF
    model_lmf = LogisticMatrixFactorization(n_factors=1)
    model_lmf.fit(matrix, epochs=10)
    assert model_lmf._is_fitted
    assert model_lmf.ideal_points_.shape == (10, 1)

def test_model_predict_proba():
    votes = np.array([
        [1, 0],
        [0, 1]
    ])
    matrix = RollCallMatrix(votes)
    model = WNominate(n_dims=1)
    model.fit(matrix, epochs=5)
    
    probs = model.predict_proba(matrix)
    # Probs should be tensor of shape (4,) corresponding to non-nan entries
    assert probs.shape == (4,)
    assert torch.all((probs >= 0) & (probs <= 1))
