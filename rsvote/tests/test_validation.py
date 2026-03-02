import pytest
import numpy as np
import torch
from rsvote.data import RollCallMatrix
from rsvote.models import WNominate

def test_cross_validation_split():
    np.random.seed(42)
    votes = np.random.choice([0, 1, np.nan], size=(20, 30), p=[0.4, 0.4, 0.2])
    matrix = RollCallMatrix(votes)
    
    train_mat, test_mat = matrix.train_test_split(test_size=0.1, random_state=123)
    
    # Check shapes
    assert train_mat.shape == matrix.shape
    assert test_mat.shape == matrix.shape
    
    # Check that they represent a partition of non-nulls
    orig_non_null = np.sum(~np.isnan(matrix.votes))
    train_non_null = np.sum(~np.isnan(train_mat.votes))
    test_non_null = np.sum(~np.isnan(test_mat.votes))
    
    assert train_non_null + test_non_null == orig_non_null
    
    # Check validation runs
    model = WNominate(n_dims=1)
    model.fit(train_mat, X_val=test_mat, epochs=20, lr=0.1)
    assert model._is_fitted
