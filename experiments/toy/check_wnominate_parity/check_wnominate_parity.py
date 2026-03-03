import os
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

try:
    import rpy2.robjects as robjects
    from rpy2.robjects import numpy2ri
    from rpy2.robjects.conversion import localconverter
    from rpy2.robjects.packages import importr
except ImportError:
    print(
        "rpy2 is required to run this experiment. Please install it with `uv run pip install rpy2`."
    )
    sys.exit(1)

from rsvote.data import RollCallMatrix
from rsvote.models import WNominate

warnings.filterwarnings("ignore")


def generate_synthetic_data(n_leg=200, n_votes=100, missing_rate=0.1, seed=42):
    """
    Generates a synthetic 1D roll call matrix using Gaussian Utility.
    """
    rng = np.random.default_rng(seed)

    # True ideal points uniformly distributed in unit interval
    true_ideal_points = rng.uniform(-1, 1, n_leg)

    # True vote midpoints and spreads
    true_midpoints = rng.uniform(-0.8, 0.8, n_votes)
    true_spreads = rng.uniform(0.3, 1.0, n_votes)

    beta = 15.0

    # Generate voting matrix based on exact Gaussian utility formulation
    votes = np.zeros((n_leg, n_votes))
    for i in range(n_leg):
        x_i = true_ideal_points[i]
        for j in range(n_votes):
            z_mid = true_midpoints[j]
            z_spread = true_spreads[j]

            z_yea = z_mid + z_spread
            z_nay = z_mid - z_spread

            # Exact Gaussian Utility Difference
            u_yea = beta * np.exp(-0.5 * (x_i - z_yea) ** 2)
            u_nay = beta * np.exp(-0.5 * (x_i - z_nay) ** 2)

            prob_yea = 1.0 / (1.0 + np.exp(-(u_yea - u_nay)))
            votes[i, j] = 1 if rng.random() < prob_yea else 0

    # Add random missingness
    mask = rng.random((n_leg, n_votes)) < missing_rate
    votes_with_nas = votes.copy()
    votes_with_nas[mask] = np.nan

    return true_ideal_points, votes_with_nas


def run_r_wnominate(votes_matrix):
    """Runs the original W-NOMINATE algorithm via R."""
    print("Running R wnominate...")
    try:
        importr("pscl")
        importr("wnominate")
    except Exception as e:
        print(f"Required R packages (pscl, wnominate) not found: {e}")
        sys.exit(1)

    with localconverter(robjects.default_converter + numpy2ri.converter):
        robjects.globalenv["votes_matrix"] = votes_matrix

    robjects.r("""
    suppressMessages(library(pscl))
    suppressMessages(library(wnominate))
    
    # Recode for pscl::rollcall (1=yea, 6=nay, 9=missing)
    rc_mat <- votes_matrix
    rc_mat[rc_mat == 0] <- 6
    rc_mat[is.na(rc_mat)] <- 9
    
    rc <- rollcall(rc_mat, yea=1, nay=6, missing=9, notInLegis=NA,
                   legis.names=paste0("L", 1:nrow(rc_mat)),
                   vote.names=paste0("V", 1:ncol(rc_mat)))
                   
    # Run wnominate silently
    capture.output({
        result <- wnominate(rc, dims=1, polarity=c(1), minvotes=5, lop=0.01)
    })
    
    ideal_points <- result$legislators$coord1D
    """)

    return np.array(robjects.globalenv["ideal_points"])


def run_python_wnominate(votes_matrix):
    """Runs our new PyTorch-based W-NOMINATE."""
    print("Running Python PyTorch W-NOMINATE (SGD)...")
    matrix = RollCallMatrix(votes_matrix)
    model = WNominate(
        n_dims=1, init_beta=15.0, epochs=1500, lr=0.05, method="gradient_descent"
    )

    torch.manual_seed(42)
    model.fit(matrix)

    return model.ideal_points_.flatten()


def run_python_alternating_wnominate(votes_matrix):
    """Runs our new PyTorch-based Alternating W-NOMINATE."""
    print("Running Python PyTorch Alternating W-NOMINATE...")
    matrix = RollCallMatrix(votes_matrix)
    model = WNominate(
        n_dims=1,
        init_beta=15.0,
        method="alternating",
        epochs=30,
        alternating_inner_steps=20,
        lr=0.05,
    )

    torch.manual_seed(42)
    model.fit(matrix)

    return model.ideal_points_.flatten()


def normalize(x):
    """Safely normalize to range [-1, 1]."""
    x_centered = x - np.nanmean(x)
    max_val = np.nanmax(np.abs(x_centered))
    if max_val == 0:
        return x_centered
    return x_centered / max_val


def plot_comparison(true_points, r_points, py_points, py_alt_points):
    """Generates comparison plots."""
    print("Generating comparison plots...")

    valid_idx = ~np.isnan(r_points) & ~np.isnan(py_points) & ~np.isnan(py_alt_points)
    true_points = true_points[valid_idx]
    r_points = r_points[valid_idx]
    py_points = py_points[valid_idx]
    py_alt_points = py_alt_points[valid_idx]

    r_norm = normalize(r_points)
    py_norm = normalize(py_points)
    py_alt_norm = normalize(py_alt_points)

    # Handle polarity
    if np.corrcoef(r_norm, py_norm)[0, 1] < 0:
        py_norm = -py_norm
    if np.corrcoef(r_norm, py_alt_norm)[0, 1] < 0:
        py_alt_norm = -py_alt_norm
    if np.corrcoef(true_points, r_norm)[0, 1] < 0:
        true_points = -true_points

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 4, figsize=(24, 6))

    # 1. R vs PyTorch scatter
    corr = np.corrcoef(r_norm, py_norm)[0, 1]
    sns.scatterplot(x=r_norm, y=py_norm, ax=axes[0], alpha=0.6, color="purple")
    axes[0].plot([-1, 1], [-1, 1], "k--", alpha=0.5)
    axes[0].set_title(f"PyTorch vs R\nCorr: {corr:.4f}")

    # 2. R vs PyTorch Alternating scatter
    corr_alt = np.corrcoef(r_norm, py_alt_norm)[0, 1]
    sns.scatterplot(x=r_norm, y=py_alt_norm, ax=axes[1], alpha=0.6, color="green")
    axes[1].plot([-1, 1], [-1, 1], "k--", alpha=0.5)
    axes[1].set_title(f"PyTorch Alternating vs R\nCorr: {corr_alt:.4f}")

    # 3. True vs PyTorch
    corr_true = np.corrcoef(true_points, py_norm)[0, 1]
    sns.scatterplot(x=true_points, y=py_norm, ax=axes[2], alpha=0.6, color="blue")
    axes[2].plot([-1, 1], [-1, 1], "k--", alpha=0.5)
    axes[2].set_title(f"PyTorch vs True\nCorr: {corr_true:.4f}")

    # 4. Distributions
    sns.kdeplot(r_norm, ax=axes[3], label="R", fill=True, alpha=0.3, color="red")
    sns.kdeplot(py_norm, ax=axes[3], label="PyTorch", fill=True, alpha=0.3, color="blue")
    sns.kdeplot(
        py_alt_norm, ax=axes[3], label="Py-Alt", fill=True, alpha=0.3, color="green"
    )
    axes[3].set_title("Point Distributions")
    axes[3].legend()

    plt.tight_layout()
    output_path = os.path.join(os.path.dirname(__file__), "wnominate_comparison.png")
    plt.savefig(output_path, dpi=300)
    print(f"Plot saved to: {output_path}")

    return r_norm, py_norm, py_alt_norm, true_points


def main():
    output_lines = []

    def log(msg):
        print(msg)
        output_lines.append(msg)

    log("--- W-NOMINATE Parity Experiment ---")

    # 1. Generate data
    log("Generating synthetic 1D roll call dataset...")
    true_ideal_points, votes_matrix = generate_synthetic_data(
        n_leg=100, n_votes=50, seed=123
    )
    log(
        f"Dataset Details: {votes_matrix.shape[0]} Legislators, {votes_matrix.shape[1]} Roll Calls."
    )

    # 2. Run models
    r_ideal_points = run_r_wnominate(votes_matrix)
    py_ideal_points = run_python_wnominate(votes_matrix)
    py_alt_ideal_points = run_python_alternating_wnominate(votes_matrix)

    # 3. Evaluate
    r_n, py_n, py_alt_n, true_pts = plot_comparison(
        true_ideal_points, r_ideal_points, py_ideal_points, py_alt_ideal_points
    )

    corr = np.corrcoef(r_n, py_n)[0, 1]
    corr_alt = np.corrcoef(r_n, py_alt_n)[0, 1]
    corr_true = np.corrcoef(true_pts, py_n)[0, 1]

    log("\n--- Pearson Correlations ---")
    log(f"R vs PyTorch: {corr:.4f}")
    log(f"R vs Py-Alt: {corr_alt:.4f}")
    log(f"True vs PyTorch: {corr_true:.4f}")

    log("\nExperiment completed successfully.")

    out_path = os.path.join(os.path.dirname(__file__), "results.txt")
    with open(out_path, "w") as f:
        f.write("\n".join(output_lines) + "\n")


if __name__ == "__main__":
    main()
