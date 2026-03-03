# RS-Vote: Roll Call Scaling & Recommendation Models

A PyTorch-accelerated package for legislative roll-call scaling, featuring an exact W-NOMINATE implementation and FM-COV (Factorization Machines with Covariates).

## Features

- **Exact W-NOMINATE**: Mathematical parity with the original Poole & Rosenthal formulation using Gaussian utility differences.
- **FM-COV**: State-of-the-art roll-call scaling using Factorization Machines with categorical covariates.
- **Logistic Matrix Factorization**: Standard collaborative filtering for binary voting data.
- **PyTorch Accelerated**: Leverages automatic differentiation and GPU support for faster convergence.
- **Scikit-learn API**: Familiar `fit()`, `predict_proba()`, and `predict()` methods.

## Installation

```bash
# Clone the repository
git clone https://github.com/vcoscrato/rs-vote.git
cd rs-vote

# Install with dependencies (using uv recommended)
uv pip install .
```

## Quick Start

### W-NOMINATE

```python
import numpy as np
from rsvote import RollCallMatrix, WNominate

# Load your voting data (Rows: Legislators, Cols: Roll Calls)
# 1 for Yea, 0 for Nay, NaN for missing
votes = np.array([
    [1, 0, 1],
    [0, 1, 0],
    # ...
])

# Initialize and fit
matrix = RollCallMatrix(votes)
model = WNominate(n_dims=2)
model.fit(matrix)

# Extract ideal points
ideal_points = model.ideal_points_
```

### FM-COV with Covariates

```python
import pandas as pd
from rsvote import RollCallMatrix, FMCov

# Legislative metadata (parties, states, etc.)
legis_df = pd.DataFrame({
    "id": range(n_legislators),
    "cov_party": ["P1", "P2", ...]
})

matrix = RollCallMatrix(votes, legislators=legis_df)
model = FMCov(n_factors=5)
model.fit(matrix)

# Effective ideal points (base + covariate shifts)
points = model.ideal_points_
```

## Mathematical Methodology

### W-NOMINATE Utility
RS-Vote uses the exact Gaussian utility difference:
$$logit = \beta \left[ \exp\left(-\frac{1}{2} d_{Yeas}^2\right) - \exp\left(-\frac{1}{2} d_{Nays}^2\right) \right]$$
where $d^2 = \sum w_k^2 (x_k - z_k)^2$.

### FM-COV
FM-COV incorporates legislative side information directly into the latent space:
$$P_{i, effective} = p_i + \sum_{m} \mathcal{C}_{i,m}$$
where $\mathcal{C}_{i,m}$ is the latent shift associated with covariate level $m$ for legislator $i$.

## Future Work

- **Early stopping**: Monitor validation loss and stop training when it plateaus or increases, preventing overfitting on small datasets.
- **Shared fit loop**: Factor out the common SGD training loop from `LogisticMatrixFactorization` and `FMCov` into a reusable base method.
- **Unit tests**: Add a proper test suite (currently only integration-level experiment scripts exist).
- **LMF with covariates**: Allow `LogisticMatrixFactorization` to accept covariates, bridging the gap with `FMCov`.
- **Proximal gradient for L1**: Replace subgradient-based L1 regularization with proximal operators (soft-thresholding) for cleaner sparsity.
- **GPU-native data pipeline**: Support `DataLoader`-based mini-batching for datasets that don't fit in memory.
- **Additional models**: Implement other ideal point methods (e.g., IRT-based models, Bayesian approaches).

## License
MIT
