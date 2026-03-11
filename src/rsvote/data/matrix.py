import numpy as np
import pandas as pd
import torch


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

    All internal representations are stored as PyTorch tensors to avoid
    redundant numpy↔tensor conversions during training.
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
            votes_np = votes.values.astype(float)
        else:
            votes_np = np.asarray(votes, dtype=float)

        self.n_legislators, self.n_votes = votes_np.shape

        # Validation
        unique_vals = np.unique(votes_np[~np.isnan(votes_np)])
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

        # ── Store as tensors directly ──────────────────────────────────
        # COO sparse representation of observed entries
        user_idx, item_idx = np.where(~np.isnan(votes_np))
        labels = votes_np[user_idx, item_idx]

        self.user_idx = torch.tensor(user_idx, dtype=torch.long)
        self.item_idx = torch.tensor(item_idx, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.float32)

        # Pre-compute covariate tensors (once, at construction)
        self.user_cov_tensors, self.user_cov_dims = self._factorize_covariates(
            self.legislators
        )
        self.item_cov_tensors, self.item_cov_dims = self._factorize_covariates(
            self.rollcalls
        )

    @staticmethod
    def _factorize_covariates(
        metadata: pd.DataFrame | None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, int]]:
        """
        Parses metadata for any columns prefixed with `cov_` and converts them
        to integer-coded LongTensors suitable for nn.Embedding.

        Returns:
            tensors: Dict mapping column name to LongTensor of integer codes
            dims: Dict mapping column name to the number of unique categories
        """
        tensors = {}
        dims = {}
        if metadata is not None:
            for col in metadata.columns:
                if str(col).startswith("cov_"):
                    codes, uniques = pd.factorize(metadata[col])
                    # Shift codes: factorize returns -1 for NaN, we shift to 0
                    tensors[col] = torch.tensor(codes + 1, dtype=torch.long)
                    dims[col] = len(uniques) + 1
        return tensors, dims

    @property
    def shape(self) -> tuple[int, int]:
        return (self.n_legislators, self.n_votes)

    def __repr__(self):
        return f"RollCallMatrix(n_legislators={self.n_legislators}, n_votes={self.n_votes})"

    def __eq__(self, other):
        if not isinstance(other, RollCallMatrix):
            return False
        return (
            torch.equal(self.user_idx, other.user_idx)
            and torch.equal(self.item_idx, other.item_idx)
            and torch.equal(self.labels, other.labels)
        )

    def to_pytorch_tensors(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns the COO sparse representation as PyTorch tensors.

        Returns:
            user_idx: LongTensor of legislator indices
            item_idx: LongTensor of vote indices
            labels: FloatTensor of binary vote outcomes (0 or 1)
        """
        return self.user_idx, self.item_idx, self.labels

    def get_user_covariate_tensors(
        self, user_idx: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Index user covariate tensors by observation-level user indices."""
        return {name: t[user_idx] for name, t in self.user_cov_tensors.items()}

    def get_item_covariate_tensors(
        self, item_idx: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Index item covariate tensors by observation-level item indices."""
        return {name: t[item_idx] for name, t in self.item_cov_tensors.items()}

    def get_all_user_covariate_tensors(self) -> dict[str, torch.Tensor]:
        """Return full user covariate tensors (one entry per legislator)."""
        return self.user_cov_tensors

    def train_test_split(self, test_size: float = 0.1, n_last_votes: int | None = None, random_state: int | None = None):
        """
        Splits the RollCallMatrix into a training and a testing set.

        Args:
            test_size: Proportion of observed votes to include in the test split.
                       Ignored if `n_last_votes` is provided.
            n_last_votes: If provided, instead of a random global split, the last
                          `n_last_votes` items for each user will be placed in the test set.
                          The method assumes item indices represent chronological order.
            random_state: Seed for the random number generator.

        Returns:
            train_matrix: A RollCallMatrix with test votes set to NaN.
            test_matrix: A RollCallMatrix containing ONLY the test votes (others NaN).
        """
        votes_np = np.full((self.n_legislators, self.n_votes), np.nan)
        u = self.user_idx.numpy()
        i = self.item_idx.numpy()
        l = self.labels.numpy()
        votes_np[u, i] = l

        n_observed = len(u)
        test_mask = np.zeros(n_observed, dtype=bool)

        if n_last_votes is not None:
            # Time-dependent split per user
            # Group observations by user
            # Note: since COO extraction goes row by row and within row column by column,
            # this is already sorted if item indices are sorted. But we sort to be safe.
            df_obs = pd.DataFrame({'u': u, 'i': i, 'idx': np.arange(n_observed)})
            
            # Sort by user then item index (assuming item index is chronological)
            df_obs = df_obs.sort_values(['u', 'i'])
            
            # For each user, pick the last N items
            # Avoid putting ALL items of a user in test: leave at least 1 item in train
            def get_test_idx(group):
                n_available = len(group)
                # Keep at least 1 item for training
                n_test = min(n_last_votes, n_available - 1)
                if n_test <= 0:
                    return pd.Series(dtype=int)
                return group['idx'].tail(n_test)
                
            test_indices = df_obs.groupby('u', as_index=False).apply(get_test_idx).values
            if len(test_indices) > 0:
                test_mask[test_indices.astype(int)] = True
                
        else:
            # Global random split
            rng = np.random.default_rng(random_state)
            indices = rng.permutation(n_observed)
            n_test = int(n_observed * test_size)
            test_indices = indices[:n_test]
            test_mask[test_indices] = True

        train_indices = ~test_mask

        # Masking for train
        train_votes = np.full_like(votes_np, np.nan)
        train_votes[u[train_indices], i[train_indices]] = l[train_indices]

        # Masking for test
        test_votes = np.full_like(votes_np, np.nan)
        test_mask_idx = np.where(test_mask)[0]
        test_votes[u[test_mask_idx], i[test_mask_idx]] = l[test_mask_idx]

        train_mat = RollCallMatrix(train_votes, self.legislators, self.rollcalls)
        test_mat = RollCallMatrix(test_votes, self.legislators, self.rollcalls)

        return train_mat, test_mat
