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


def generate_synthetic_2d_data(
    n_leg=200, n_votes=150, missing_rate=0.1, beta=15.0, seed=42
):
    """
    Generates a synthetic 2D roll call matrix using Exact W-NOMINATE math.
    """
    rng = np.random.default_rng(seed)

    # True ideal points uniformly distributed in the unit circle
    r = rng.uniform(0, 1, n_leg)
    theta = rng.uniform(0, 2 * np.pi, n_leg)
    true_ideal_points = np.zeros((n_leg, 2))
    true_ideal_points[:, 0] = r * np.cos(theta)
    true_ideal_points[:, 1] = r * np.sin(theta)

    # True dimension weights: w1=1, w2=0.5
    dim_weights = np.array([1.0, 0.5])
    w_sq = dim_weights**2

    # True vote midpoints and spreads
    true_midpoints = rng.uniform(-0.8, 0.8, (n_votes, 2))
    true_spreads = rng.uniform(0.3, 0.8, (n_votes, 2))

    votes = np.zeros((n_leg, n_votes))
    for i in range(n_leg):
        x_i = true_ideal_points[i]
        for j in range(n_votes):
            z_mid = true_midpoints[j]
            z_spread = true_spreads[j]

            z_yea = z_mid + z_spread
            z_nay = z_mid - z_spread

            # Exact Gaussian Utility Difference with weights
            d_yea_sq = (w_sq * (x_i - z_yea) ** 2).sum()
            d_nay_sq = (w_sq * (x_i - z_nay) ** 2).sum()

            logit = beta * (np.exp(-0.5 * d_yea_sq) - np.exp(-0.5 * d_nay_sq))
            prob_yea = 1.0 / (1.0 + np.exp(-logit))
            votes[i, j] = 1 if rng.random() < prob_yea else 0

    # Add random missingness
    mask = rng.random((n_leg, n_votes)) < missing_rate
    votes_with_nas = votes.copy()
    votes_with_nas[mask] = np.nan

    return true_ideal_points, votes_with_nas


def run_r_wnominate_2d(votes_matrix, true_ideal_points):
    """Runs the original W-NOMINATE algorithm via R for 2 Dimensions."""
    print("Running R wnominate (2D)...")
    try:
        importr("pscl")
        importr("wnominate")
    except Exception as e:
        print(f"Required R packages (pscl, wnominate) not found: {e}")
        sys.exit(1)

    # Polarity anchors
    leg_d1_pos_idx = np.argmax(true_ideal_points[:, 0]) + 1
    leg_d2_pos_idx = np.argmax(true_ideal_points[:, 1]) + 1

    with localconverter(robjects.default_converter + numpy2ri.converter):
        robjects.globalenv["votes_matrix"] = votes_matrix
        robjects.globalenv["leg_d1_pos_idx"] = leg_d1_pos_idx
        robjects.globalenv["leg_d2_pos_idx"] = leg_d2_pos_idx

    robjects.r("""
    suppressMessages(library(pscl))
    suppressMessages(library(wnominate))
    
    rc_mat <- votes_matrix
    rc_mat[rc_mat == 0] <- 6
    rc_mat[is.na(rc_mat)] <- 9
    
    rc <- rollcall(rc_mat, yea=1, nay=6, missing=9, notInLegis=NA,
                   legis.names=paste0("L", 1:nrow(rc_mat)),
                   vote.names=paste0("V", 1:ncol(rc_mat)))
                   
    capture.output({
        result <- wnominate(rc, dims=2, polarity=c(leg_d1_pos_idx, leg_d2_pos_idx), minvotes=5, lop=0.01)
    })
    
    ideal_d1 <- result$legislators$coord1D
    ideal_d2 <- result$legislators$coord2D
    """)

    return np.column_stack(
        (
            np.array(robjects.globalenv["ideal_d1"]),
            np.array(robjects.globalenv["ideal_d2"]),
        )
    )


def run_python_wnominate_2d(votes_matrix, use_cv=False):
    """Runs our PyTorch-based W-NOMINATE in 2D."""
    print(f"Running Python W-NOMINATE (2D) - CV: {use_cv}...")
    matrix = RollCallMatrix(votes_matrix)

    model = WNominate(
        n_dims=2, init_beta=15.0, epochs=2000, lr=0.05, method="gradient_descent"
    )

    if use_cv:
        train_mat, val_mat = matrix.train_test_split(test_size=0.1, random_state=42)
        model.fit(train_mat, X_val=val_mat)
    else:
        model.fit(matrix)

    return model.ideal_points_


def procrustes_alignment(target_pts, source_pts):
    """Align source_pts to target_pts via rotation/reflection/scale."""
    # Filter NaNs
    valid_idx = ~np.isnan(target_pts).any(axis=1) & ~np.isnan(source_pts).any(axis=1)
    A = target_pts[valid_idx]
    B = source_pts[valid_idx]

    # Center
    mu_A = A.mean(axis=0)
    mu_B = B.mean(axis=0)
    A_c = A - mu_A
    B_c = B - mu_B

    # Normalize scales (Frobenius)
    norm_A = np.linalg.norm(A_c)
    norm_B = np.linalg.norm(B_c)
    if norm_B > 0:
        A_c = A_c / norm_A
        B_c = B_c / norm_B

    # Orthogonal Procrustes
    U, _, Vt = np.linalg.svd(B_c.T @ A_c)
    R = U @ Vt

    # Apply to centered+scaled B
    B_aligned = B_c @ R
    
    # Scale back to A's scale
    B_final = B_aligned * norm_A
    A_final = A_c * norm_A
    
    return A_final, B_final


def plot_2d_comparisons(true_pts, r_pts, py_no_cv, py_cv, out_lines):
    print("Generating 2D comparison plots...")

    # Align each estimate to truth
    true_r, r_aligned = procrustes_alignment(true_pts, r_pts)
    true_py, py_aligned = procrustes_alignment(true_pts, py_no_cv)
    true_cv, cv_aligned = procrustes_alignment(true_pts, py_cv)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    sns.set_theme(style="whitegrid")

    def plot_scatter(ax, target, source, title, color):
        ax.scatter(target[:, 0], target[:, 1], c="gray", alpha=0.2, label="True")
        ax.scatter(source[:, 0], source[:, 1], c=color, alpha=0.7, marker="x", label="Est")
        
        # Displacement lines
        for i in range(len(target)):
            ax.plot([target[i,0], source[i,0]], [target[i,1], source[i,1]], 
                    c="gray", alpha=0.1, linewidth=0.5)

        ax.set_title(title)
        ax.axis("equal")
        
        mse = np.linalg.norm(target - source) / len(target)
        c1 = np.corrcoef(target[:,0], source[:,0])[0,1]
        c2 = np.corrcoef(target[:,1], source[:,1])[0,1]
        
        ax.text(0.05, 0.95, f"MSE: {mse:.4f}\nD1 Corr: {c1:.3f}\nD2 Corr: {c2:.3f}", 
                transform=ax.transAxes, verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
        
        out_lines.append(f"{title} -> MSE: {mse:.4f} | D1 Corr: {c1:.3f} | D2 Corr: {c2:.3f}")

    plot_scatter(axes[0], true_r, r_aligned, "R W-NOMINATE", "red")
    plot_scatter(axes[1], true_py, py_aligned, "Python W-NOMINATE", "blue")
    plot_scatter(axes[2], true_cv, cv_aligned, "Python W-NOMINATE (CV)", "purple")

    plt.tight_layout()
    output_path = os.path.join(os.path.dirname(__file__), "wnominate_2d_comparison.png")
    plt.savefig(output_path, dpi=300)
    print(f"Plot saved to: {output_path}")
    return output_path


def main():
    out_lines = []
    def log(msg):
        print(msg)
        out_lines.append(msg)

    log("--- 2D W-NOMINATE Recovery Experiment ---")

    true_ideal_points, votes_matrix = generate_synthetic_2d_data(
        n_leg=150, n_votes=100, seed=42
    )

    r_pts = run_r_wnominate_2d(votes_matrix, true_ideal_points)
    py_pts = run_python_wnominate_2d(votes_matrix, use_cv=False)
    py_cv_pts = run_python_wnominate_2d(votes_matrix, use_cv=True)

    plot_path = plot_2d_comparisons(true_ideal_points, r_pts, py_pts, py_cv_pts, out_lines)
    log(f"\nSaved plots to: {plot_path}")
    
    out_path = os.path.join(os.path.dirname(__file__), "results.txt")
    with open(out_path, "w") as f:
        f.write("\n".join(out_lines) + "\n")


if __name__ == "__main__":
    main()
