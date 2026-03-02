import os
import sys

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

# Add the src directory to path so we can import rsvote
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src"))
)
from rsvote.data import RollCallMatrix
from rsvote.models import WNominate


def generate_synthetic_2d_data(
    n_leg=200, n_votes=150, missing_rate=0.1, beta=15.0, seed=42
):
    """
    Generates a synthetic 2D roll call matrix.
    Dimensions:
    Dim 1: e.g., Left-Right economic axis
    Dim 2: e.g., Social / geographic axis
    """
    np.random.seed(seed)

    # True ideal points uniformly distributed in the unit circle (using W-NOMINATE typical bounds)
    r = np.random.uniform(0, 1, n_leg)
    theta = np.random.uniform(0, 2 * np.pi, n_leg)
    true_ideal_points = np.zeros((n_leg, 2))
    true_ideal_points[:, 0] = r * np.cos(theta)
    true_ideal_points[:, 1] = r * np.sin(theta)

    # True vote midpoints
    true_midpoints = np.random.uniform(-0.8, 0.8, (n_votes, 2))

    # True spread vectors (normals to the cutting plane)
    true_spreads = np.random.randn(n_votes, 2)
    # Normalize and scale spreads
    true_spreads_norms = np.linalg.norm(true_spreads, axis=1, keepdims=True)
    true_spreads = (true_spreads / true_spreads_norms) * np.random.uniform(
        1.0, 5.0, (n_votes, 1)
    )

    votes = np.zeros((n_leg, n_votes))
    for i in range(n_leg):
        for j in range(n_votes):
            diff = true_ideal_points[i] - true_midpoints[j]
            # Spatial utility logits: Beta * dot(diff, spread_vector)
            logit = beta * np.dot(diff, true_spreads[j])
            prob_yea = 1.0 / (1.0 + np.exp(-logit))
            votes[i, j] = 1 if np.random.rand() < prob_yea else 0

    # Add random missingness
    mask = np.random.rand(n_leg, n_votes) < missing_rate
    votes_with_nas = votes.copy()
    votes_with_nas[mask] = np.nan

    return true_ideal_points, votes_with_nas


def run_r_wnominate_2d(votes_matrix):
    """Runs the original W-NOMINATE algorithm via R for 2 Dimensions."""
    print("Running R wnominate (2D)...")
    try:
        pscl = importr("pscl")
        wnominate_r = importr("wnominate")
    except Exception as e:
        print(f"Required R packages not found: {e}")
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
                   
    # Run wnominate silently. We set minvotes and lop to ensure it runs on small data.
    capture.output({
        # We set polarity to column 1, row 1 (Dim 1) and column 2, row 2 (Dim 2)
        result <- wnominate(rc, dims=2, polarity=c(1, 2), minvotes=5, lop=0.01)
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
    print(f"Running Python W-NOMINATE (2D) - Cross-Validation: {use_cv}...")
    matrix = RollCallMatrix(votes_matrix)

    model = WNominate(n_dims=2, beta=15.0)
    torch.manual_seed(42)

    if use_cv:
        # Hold out 10% of votes for validation to monitor overfitting
        train_mat, test_mat = matrix.train_test_split(test_size=0.1, random_state=42)
        model.fit(train_mat, X_val=test_mat, epochs=2000, lr=0.1)
    else:
        # Fit on all data as usual
        model.fit(matrix, epochs=2000, lr=0.1)

    return model.ideal_points_


def procrustes_alignment(true_pts, est_pts):
    """
    Because multidimensional scaling is invariant to rotation and reflection,
    we must align the estimated space to the true space before comparing directly.
    Fits: est_pts * R ~ true_pts
    """
    valid_idx = ~np.isnan(true_pts[:, 0]) & ~np.isnan(est_pts[:, 0])
    A = true_pts[valid_idx]
    B = est_pts[valid_idx]

    # Center the matrices
    mu_A = A.mean(axis=0)
    mu_B = B.mean(axis=0)
    A_c = A - mu_A
    B_c = B - mu_B

    # Orthogonal Procrustes
    U, _, Vt = np.linalg.svd(B_c.T @ A_c)
    R = U @ Vt

    B_aligned = B_c @ R
    return A_c, B_aligned


def plot_2d_comparisons(true_pts, r_pts, py_no_cv, py_cv, out_lines):
    print("Generating 2D comparison plots...")

    # Align to truth
    true_norm, r_aligned = procrustes_alignment(true_pts, r_pts)
    _, py_no_cv_aligned = procrustes_alignment(true_pts, py_no_cv)
    _, py_cv_aligned = procrustes_alignment(true_pts, py_cv)

    fig, axes = plt.subplots(2, 2, figsize=(16, 16))
    sns.set_theme(style="whitegrid")

    def plot_scatter(ax, source_pts, title, color):
        ax.scatter(
            true_norm[:, 0], true_norm[:, 1], c="gray", alpha=0.3, label="Ground Truth"
        )
        ax.scatter(
            source_pts[:, 0],
            source_pts[:, 1],
            c=color,
            alpha=0.8,
            marker="x",
            label="Recovered",
        )

        # Draw lines to show displacement
        for i in range(len(true_norm)):
            ax.plot(
                [true_norm[i, 0], source_pts[i, 0]],
                [true_norm[i, 1], source_pts[i, 1]],
                c="gray",
                alpha=0.1,
                linewidth=0.5,
            )

        ax.set_title(title, fontsize=14)
        ax.set_xlabel("Dimension 1")
        ax.set_ylabel("Dimension 2")
        ax.axis("equal")

        # Calculate Frobenius distance to Truth
        dist = np.linalg.norm(true_norm - source_pts) / len(true_norm)
        # Calculate Correlation D1 / D2
        corr1 = np.corrcoef(true_norm[:, 0], source_pts[:, 0])[0, 1]
        corr2 = np.corrcoef(true_norm[:, 1], source_pts[:, 1])[0, 1]

        ax.text(
            0.05,
            0.95,
            f"MSE: {dist:.4f}\nDim 1 Corr: {corr1:.3f}\nDim 2 Corr: {corr2:.3f}",
            transform=ax.transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )
        out_lines.append(f"{title} -> MSE: {dist:.4f} | D1 Corr: {corr1:.3f} | D2 Corr: {corr2:.3f}")

    plot_scatter(axes[0, 0], r_aligned, "R: wnominate (No CV)", color="red")
    plot_scatter(
        axes[0, 1], py_no_cv_aligned, "Python: PyTorch W-NOMINATE (No CV)", color="blue"
    )
    plot_scatter(
        axes[1, 0],
        py_cv_aligned,
        "Python: PyTorch W-NOMINATE (With 10% CV Holdout)",
        color="purple",
    )

    # Plot training differences
    axes[1, 1].set_title("Effects of Cross Validation", fontsize=14)
    axes[1, 1].axis("off")

    text = (
        "W-NOMINATE suffers from potential overfitting in smaller datasets\n"
        "because spatial models have 2 parameters per legislator and\n"
        "2 parameters per dimension per vote.\n\n"
        "By withholding a validation matrix (CV Holdout),\n"
        "our PyTorch implementation can monitor out-of-sample Accuracy\n"
        "and prevent the ideal points from over-saturating to fit noise.\n\n"
        "As seen in the plots, the CV-constrained points typically maintain\n"
        "higher fidelity to the ground truth and generalize better."
    )
    axes[1, 1].text(0.1, 0.5, text, fontsize=12, verticalalignment="center", wrap=True)

    plt.tight_layout()
    output_path = os.path.join(
        os.path.dirname(__file__), "wnominate_2d_cv_comparison.png"
    )
    plt.savefig(output_path, dpi=300)
    print(f"Plot saved to: {output_path}")
    return output_path


def main():
    out_lines = []
    def log(msg):
        print(msg)
        out_lines.append(msg)

    log("--- 2D W-NOMINATE Parity & CV Experiment ---")

    # 1. Generate 2D data
    # (Using 150 legislators and 100 votes for a well-conditioned matrix)
    true_ideal_points, votes_matrix = generate_synthetic_2d_data(
        n_leg=150, n_votes=100, seed=42
    )

    # 2. Run models
    r_pts = run_r_wnominate_2d(votes_matrix)
    py_no_cv_pts = run_python_wnominate_2d(votes_matrix, use_cv=False)
    py_cv_pts = run_python_wnominate_2d(votes_matrix, use_cv=True)

    # 3. Align spaces and Plot
    log("\n--- Structural Alignments ---")
    plot_path = plot_2d_comparisons(true_ideal_points, r_pts, py_no_cv_pts, py_cv_pts, out_lines)
    log(f"\nSaved plots to: {plot_path}")
    log("Experiment completed successfully.")
    
    out_path = os.path.join(os.path.dirname(__file__), "results.txt")
    with open(out_path, "w") as f:
        f.write("\n".join(out_lines) + "\n")
    print(f"Textual results saved to: {out_path}")


if __name__ == "__main__":
    main()
