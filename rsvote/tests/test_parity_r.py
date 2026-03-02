import pytest
import numpy as np
import pandas as pd
import torch

pytest.importorskip("rpy2")
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter

from rsvote.data import RollCallMatrix
from rsvote.models import WNominate

# Ensure wnominate is installed in R
try:
    wnominate_r = importr('wnominate')
    pscl = importr('pscl')
except Exception as e:
    pytest.skip(allow_module_level=True, reason=f"Could not load R packages wnominate or pscl: {e}")

def test_r_parity_synthetic_1d():
    """
    Generate synthetic 1D data.
    Run through R wnominate and our PyTorch WNominate.
    Compare ideal points.
    """
    np.random.seed(123)
    n_leg = 100
    n_votes = 50
    
    # True ideal points uniformly distributed in [-1, 1]
    true_ideal_points = np.random.uniform(-1, 1, n_leg)
    
    # True vote midpoints
    true_midpoints = np.random.uniform(-0.8, 0.8, n_votes)
    true_spreads = np.random.uniform(2.0, 5.0, n_votes) * np.random.choice([-1, 1], n_votes)
    
    # Generate voting matrix based on spatial utility formulation
    votes = np.zeros((n_leg, n_votes))
    for i in range(n_leg):
        for j in range(n_votes):
            diff = true_ideal_points[i] - true_midpoints[j]
            logit = 15.0 * (diff * true_spreads[j]) # beta=15.0
            prob_yea = 1.0 / (1.0 + np.exp(-logit))
            votes[i, j] = 1 if np.random.rand() < prob_yea else 0
            
    # Add some random missingness
    mask = np.random.rand(n_leg, n_votes) < 0.1
    votes_with_nas = votes.copy()
    votes_with_nas[mask] = np.nan
    
    # 1. Run R wnominate
    # we need to create a pscl rollcall object first
    pscl = importr('pscl')
    wnominate_r = importr('wnominate')
    
    from rpy2.robjects import numpy2ri
    
    from rpy2.robjects import numpy2ri
    
    with localconverter(robjects.default_converter + numpy2ri.converter):
        robjects.globalenv['votes_matrix'] = votes_with_nas
        
    robjects.r('''
    library(pscl)
    library(wnominate)
    rc_mat <- votes_matrix
    rc_mat[rc_mat == 0] <- 6
    rc_mat[is.na(rc_mat)] <- 9
    rc <- rollcall(rc_mat, yea=1, nay=6, missing=9, notInLegis=NA)
    result <- wnominate(rc, dims=1, polarity=c(1))
    ideal_points <- result$legislators$coord1D
    ''')
    
    r_ideal_points = np.array(robjects.globalenv['ideal_points'])
    
    # 2. Run Python WNominate
    matrix = RollCallMatrix(votes_with_nas)
    model = WNominate(n_dims=1, beta=15.0)
    
    # Seed torch for reproducibility
    torch.manual_seed(42)
    model.fit(matrix, epochs=1500, lr=0.1)
    py_ideal_points = model.ideal_points_.flatten()
    
    # Alignment: W-NOMINATE suffers from rotational/polarity invariance
    # We test absolute correlation since the scale / polarity might be flipped
    corr = np.abs(np.corrcoef(r_ideal_points[~np.isnan(r_ideal_points)], 
                               py_ideal_points[~np.isnan(r_ideal_points)])[0, 1])
    
    assert corr > 0.90, f"Correlation with R wnominate is too low: {corr:.3f}"
