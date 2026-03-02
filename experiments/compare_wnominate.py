import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch

try:
    import rpy2.robjects as robjects
    from rpy2.robjects.packages import importr
    from rpy2.robjects import numpy2ri
    from rpy2.robjects.conversion import localconverter
except ImportError:
    print("rpy2 is required to run this experiment. Please install it with `uv run pip install rpy2`.")
    sys.exit(1)

# Add the src directory to path so we can import rsvote
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../rsvote/src')))
from rsvote.data import RollCallMatrix
from rsvote.models import WNominate, AlternatingWNominate

def generate_synthetic_data(n_leg=200, n_votes=100, missing_rate=0.1, seed=42):
    """
    Generates a synthetic roll call matrix for testing.
    Uses the underlying spatial utility math of W-NOMINATE.
    """
    np.random.seed(seed)
    
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
            
    # Add random missingness
    mask = np.random.rand(n_leg, n_votes) < missing_rate
    votes_with_nas = votes.copy()
    votes_with_nas[mask] = np.nan
    
    return true_ideal_points, votes_with_nas

def run_r_wnominate(votes_matrix):
    """Runs the original W-NOMINATE algorithm via R."""
    print("Running R wnominate...")
    try:
        pscl = importr('pscl')
        wnominate_r = importr('wnominate')
    except Exception as e:
        print(f"Required R packages not found: {e}")
        sys.exit(1)
        
    with localconverter(robjects.default_converter + numpy2ri.converter):
        robjects.globalenv['votes_matrix'] = votes_matrix
        
    robjects.r('''
    suppressMessages(library(pscl))
    suppressMessages(library(wnominate))
    
    # Recode for pscl::rollcall (1=yea, 6=nay, 9=missing)
    rc_mat <- votes_matrix
    rc_mat[rc_mat == 0] <- 6
    rc_mat[is.na(rc_mat)] <- 9
    
    rc <- rollcall(rc_mat, yea=1, nay=6, missing=9, notInLegis=NA,
                   legis.names=paste0("L", 1:nrow(rc_mat)),
                   vote.names=paste0("V", 1:ncol(rc_mat)))
                   
    # Run wnominate silently, relaxing the strict filters for our synthetic data
    capture.output({
        result <- wnominate(rc, dims=1, polarity=c(1), minvotes=5, lop=0.01)
    })
    
    ideal_points <- result$legislators$coord1D
    ''')
    
    return np.array(robjects.globalenv['ideal_points'])

def run_python_wnominate(votes_matrix):
    """Runs our new PyTorch-based W-NOMINATE."""
    print("Running Python PyTorch W-NOMINATE...")
    matrix = RollCallMatrix(votes_matrix)
    model = WNominate(n_dims=1, beta=15.0)
    
    # Set seed for reproducibility
    torch.manual_seed(42)
    model.fit(matrix, epochs=1500, lr=0.1)
    
    return model.ideal_points_.flatten()
def run_python_alternating_wnominate(votes_matrix):
    """Runs our new PyTorch-based Alternating W-NOMINATE."""
    print("Running Python PyTorch Alternating W-NOMINATE...")
    matrix = RollCallMatrix(votes_matrix)
    model = AlternatingWNominate(n_dims=1, beta=15.0)
    
    # Set seed for reproducibility
    torch.manual_seed(42)
    model.fit(matrix, epochs=1500, lr=0.1)
    
    return model.ideal_points_.flatten()

def plot_comparison(true_points, r_points, py_points, py_alt_points):
    """Generates comparison plots and saves them."""
    print("Generating comparison plots...")
    
    # Filter out NaNs if any models completely failed to scale a legislator
    valid_idx = ~np.isnan(r_points) & ~np.isnan(py_points) & ~np.isnan(py_alt_points)
    true_points = true_points[valid_idx]
    r_points = r_points[valid_idx]
    py_points = py_points[valid_idx]
    py_alt_points = py_alt_points[valid_idx]
    
    # Normalize scales to [-1, 1] for visual comparison 
    # (since W-NOMINATE is invariant to rotation/scale)
    def normalize(x):
        # Center and scale
        x = x - np.mean(x)
        return x / np.max(np.abs(x))
        
    r_norm = normalize(r_points)
    py_norm = normalize(py_points)
    py_alt_norm = normalize(py_alt_points)
    
    # Ensure they have the same polarity (deal with rotation invariance)
    if np.corrcoef(r_norm, py_norm)[0, 1] < 0:
        py_norm = -py_norm
        
    if np.corrcoef(r_norm, py_alt_norm)[0, 1] < 0:
        py_alt_norm = -py_alt_norm
        
    if np.corrcoef(true_points, r_norm)[0, 1] < 0:
        true_points = -true_points

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 4, figsize=(24, 5))
    
    # 1. R vs PyTorch scatter
    corr = np.corrcoef(r_norm, py_norm)[0, 1]
    sns.scatterplot(x=r_norm, y=py_norm, ax=axes[0], alpha=0.6, color="purple")
    axes[0].plot([-1, 1], [-1, 1], 'k--', alpha=0.5)
    axes[0].set_title(f"PyTorch vs R W-NOMINATE\nCorrelation: {corr:.4f}")
    axes[0].set_xlabel("R wnominate Ideal Points (Normalized)")
    axes[0].set_ylabel("PyTorch W-NOMINATE Ideal Points (Normalized)")
    
    # 2. R vs PyTorch Alternating scatter
    corr_alt = np.corrcoef(r_norm, py_alt_norm)[0, 1]
    sns.scatterplot(x=r_norm, y=py_alt_norm, ax=axes[1], alpha=0.6, color="green")
    axes[1].plot([-1, 1], [-1, 1], 'k--', alpha=0.5)
    axes[1].set_title(f"PyTorch Alternating vs R W-NOMINATE\nCorrelation: {corr_alt:.4f}")
    axes[1].set_xlabel("R wnominate Ideal Points (Normalized)")
    axes[1].set_ylabel("PyTorch Alternating Ideal Points (Normalized)")
    
    # 3. True vs PyTorch
    corr_py_true = np.corrcoef(true_points, py_norm)[0, 1]
    sns.scatterplot(x=true_points, y=py_norm, ax=axes[2], alpha=0.6, color="blue")
    axes[2].plot([-1, 1], [-1, 1], 'k--', alpha=0.5)
    axes[2].set_title(f"PyTorch vs True Underlying Model\nCorrelation: {corr_py_true:.4f}")
    axes[2].set_xlabel("True Ideal Points")
    axes[2].set_ylabel("PyTorch W-NOMINATE Ideal Points")
    
    # 4. Kernel Density Distributions
    sns.kdeplot(r_norm, ax=axes[3], label="R wnominate", fill=True, alpha=0.3, color="red")
    sns.kdeplot(py_norm, ax=axes[3], label="PyTorch W-NOMINATE", fill=True, alpha=0.3, color="blue")
    sns.kdeplot(py_alt_norm, ax=axes[3], label="PyTorch Alternating", fill=True, alpha=0.3, color="green")
    axes[3].set_title("Distribution of Ideal Points")
    axes[3].set_xlabel("Ideal Point (Normalized)")
    axes[3].legend()
    
    plt.tight_layout()
    output_path = os.path.join(os.path.dirname(__file__), "wnominate_comparison.png")
    plt.savefig(output_path, dpi=300)
    print(f"Plot saved to: {output_path}")
    
    return output_path

def main():
    print("--- W-NOMINATE Parity Experiment ---")
    
    # 1. Generate data
    print("Generating synthetic 1D roll call dataset...")
    true_ideal_points, votes_matrix = generate_synthetic_data(n_leg=100, n_votes=50, seed=123)
    print(f"Dataset Details: {votes_matrix.shape[0]} Legislators, {votes_matrix.shape[1]} Roll Calls.")
    
    # 2. Run models
    r_ideal_points = run_r_wnominate(votes_matrix)
    py_ideal_points = run_python_wnominate(votes_matrix)
    py_alt_ideal_points = run_python_alternating_wnominate(votes_matrix)
    
    # 3. Evaluate and plot
    plot_comparison(true_ideal_points, r_ideal_points, py_ideal_points, py_alt_ideal_points)
    
    print("Experiment completed successfully.")

if __name__ == "__main__":
    main()
