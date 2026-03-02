import pytest
import numpy as np
from rsvote.data import RollCallMatrix
from rsvote.models import AlternatingWNominate

def test_alternating_fit():
    np.random.seed(42)
    votes = np.random.choice([0, 1, np.nan], size=(20, 30), p=[0.4, 0.4, 0.2])
    matrix = RollCallMatrix(votes)
    
    model = AlternatingWNominate(n_dims=1)
    model.fit(matrix, epochs=25, lr=0.1)
    
    assert model._is_fitted
    assert model.ideal_points_.shape == (20, 1)

