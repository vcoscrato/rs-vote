import numpy as np
import pandas as pd


class RollCallMatrix:
    """
    A unified data structure for handling legislative roll call data,
    similar to a Recommender Systems ratings matrix.

    Rows correspond to Legislators (users).
    Columns correspond to Votes (items).

    Valid values in the matrix:
        1: Yea
        0: Nay
       NaN: Missing / Abstain
    """

    def __init__(
        self,
        votes: np.ndarray | pd.DataFrame,
        legislators: pd.DataFrame | None = None,
        rollcalls: pd.DataFrame | None = None,
    ):
        """
        Initialize the RollCallMatrix.

        Args:
            votes: A 2D array or DataFrame of votes (shape: n_legislators x n_votes)
                   Values must be 0, 1, or NaN.
            legislators: Metadata for legislators (shape: n_legislators x n_features)
            rollcalls: Metadata for roll calls (shape: n_votes x n_features)
        """
        if isinstance(votes, pd.DataFrame):
            self.votes = votes.values
        else:
            self.votes = np.asarray(votes, dtype=float)

        self.n_legislators, self.n_votes = self.votes.shape

        # Validation
        unique_vals = np.unique(self.votes[~np.isnan(self.votes)])
        if not np.all(np.isin(unique_vals, [0, 1])):
            raise ValueError("Vote matrix must contain only 0, 1, or NaN.")

        self.legislators = legislators
        self.rollcalls = rollcalls

        if self.legislators is not None and len(self.legislators) != self.n_legislators:
            raise ValueError(
                "Legislators metadata must match the number of rows in votes."
            )

        if self.rollcalls is not None and len(self.rollcalls) != self.n_votes:
            raise ValueError(
                "Rollcalls metadata must match the number of columns in votes."
            )

    @property
    def shape(self):
        return self.n_legislators, self.n_votes

    def to_pytorch_tensors(self):
        """
        Converts the matrix to PyTorch tensors suitable for model training.
        Returns:
            user_idx: Tensor of legislator indices
            item_idx: Tensor of vote indices
            labels: Tensor of binary vote outcomes (0 or 1)
        """
        import torch

        # Get indices of non-NaN values
        user_idx, item_idx = np.where(~np.isnan(self.votes))
        labels = self.votes[user_idx, item_idx]

        return (
            torch.tensor(user_idx, dtype=torch.long),
            torch.tensor(item_idx, dtype=torch.long),
            torch.tensor(labels, dtype=torch.float32),
        )

    def train_test_split(self, test_size: float = 0.1, random_state: int | None = None):
        """
        Splits the RollCallMatrix into a training and a testing set by masking out
        a percentage of the observed votes.

        Args:
            test_size: Proportion of observed votes to include in the test split.
            random_state: Seed for the random number generator.

        Returns:
            train_matrix: A RollCallMatrix with test votes set to NaN.
            test_matrix: A RollCallMatrix containing ONLY the test votes (others NaN).
        """
        if random_state is not None:
            np.random.seed(random_state)

        # Identify all observed (non-NaN) indices
        user_idx, item_idx = np.where(~np.isnan(self.votes))
        n_observed = len(user_idx)

        # Shuffle and split
        indices = np.random.permutation(n_observed)
        n_test = int(n_observed * test_size)

        test_indices = indices[:n_test]
        train_indices = indices[n_test:]

        # Create Train Matrix
        train_votes = np.full_like(self.votes, np.nan)
        train_u = user_idx[train_indices]
        train_i = item_idx[train_indices]
        votes_arr = np.asarray(self.votes)
        train_votes[train_u, train_i] = votes_arr[train_u, train_i]

        # Create Test Matrix
        test_votes = np.full_like(self.votes, np.nan)
        test_u = user_idx[test_indices]
        test_i = item_idx[test_indices]
        test_votes[test_u, test_i] = votes_arr[test_u, test_i]

        train_mat = RollCallMatrix(train_votes, self.legislators, self.rollcalls)
        test_mat = RollCallMatrix(test_votes, self.legislators, self.rollcalls)

        return train_mat, test_mat

    def get_user_covariates(self) -> tuple[dict[str, np.ndarray], dict[str, int]]:
        """
        Parses `legislators` metadata for any columns prefixed with `cov_`.
        Returns:
            covs: Dict mapping column name to integer encoded arrays suitable for nn.Embedding
            dims: Dict mapping column name to the number of unique categories
        """
        covs = {}
        dims = {}
        if self.legislators is not None:
            for col in self.legislators.columns:
                if str(col).startswith("cov_"):
                    # Factorize missing to -1, shift to 0
                    codes, uniques = pd.factorize(self.legislators[col])
                    covs[col] = codes + 1
                    dims[col] = len(uniques) + 1
        return covs, dims

    def get_item_covariates(self) -> tuple[dict[str, np.ndarray], dict[str, int]]:
        """
        Parses `rollcalls` metadata for any columns prefixed with `cov_`.
        Returns:
            covs: Dict mapping column name to integer encoded arrays suitable for nn.Embedding
            dims: Dict mapping column name to the number of unique categories
        """
        covs = {}
        dims = {}
        if self.rollcalls is not None:
            for col in self.rollcalls.columns:
                if str(col).startswith("cov_"):
                    # Factorize missing to -1, shift to 0
                    codes, uniques = pd.factorize(self.rollcalls[col])
                    covs[col] = codes + 1
                    dims[col] = len(uniques) + 1
        return covs, dims
