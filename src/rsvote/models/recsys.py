import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm.auto import tqdm

from ..data.matrix import RollCallMatrix
from .base import BaseRollCallModel


class LogisticMatrixFactorization(BaseRollCallModel):
    """
    A standard Logistic Matrix Factorization representation.
    Treats Roll Call voting purely as a Collaborative Filtering problem
    without strict spatial utility assumptions.

    Probability of Yea = sigmoid( global_bias + bias_user + bias_item + dot(embed_user, embed_item) )
    """

    def __init__(
        self,
        n_factors: int = 2,
        use_bias: bool = True,
        epochs: int = 1000,
        lr: float = 0.05,
        verbose: bool = True,
    ):
        super().__init__()
        self.n_factors = n_factors
        self.use_bias = use_bias
        self.epochs = epochs
        self.lr = lr
        self.verbose = verbose

    def _initialize_parameters(self, n_users: int, n_items: int):
        if self.n_factors > 0:
            self.user_embedding = nn.Embedding(n_users, self.n_factors)
            self.item_embedding = nn.Embedding(n_items, self.n_factors)
            nn.init.normal_(self.user_embedding.weight, std=0.1)
            nn.init.normal_(self.item_embedding.weight, std=0.1)

        self.global_bias = nn.Parameter(torch.zeros(1))

        if self.use_bias:
            self.user_bias = nn.Embedding(n_users, 1)
            self.item_bias = nn.Embedding(n_items, 1)
            nn.init.zeros_(self.user_bias.weight)
            nn.init.zeros_(self.item_bias.weight)

        self.to(self.device)

    def forward(self, user_idx: torch.Tensor, item_idx: torch.Tensor) -> torch.Tensor:
        """Calculates the logit (pre-sigmoid) for the given user-item pairs."""
        logit = self.global_bias.squeeze()

        if self.use_bias:
            b_u = self.user_bias(user_idx).squeeze(-1)
            b_i = self.item_bias(item_idx).squeeze(-1)
            logit = logit + b_u + b_i

        if self.n_factors > 0:
            u = self.user_embedding(user_idx)
            i = self.item_embedding(item_idx)
            dot = (u * i).sum(dim=1)
            logit = logit + dot

        return logit

    def fit(
        self, X: RollCallMatrix, X_val: RollCallMatrix | None = None
    ) -> "LogisticMatrixFactorization":
        n_users, n_items = X.shape
        self._initialize_parameters(n_users, n_items)

        user_idx, item_idx, labels = X.to_pytorch_tensors()
        user_idx = user_idx.to(self.device)
        item_idx = item_idx.to(self.device)
        labels = labels.to(self.device)

        val_data = None
        if X_val is not None:
            val_user_idx, val_item_idx, val_labels = X_val.to_pytorch_tensors()
            val_data = (
                val_user_idx.to(self.device),
                val_item_idx.to(self.device),
                val_labels.to(self.device),
            )

        optimizer = Adam(self.parameters(), lr=self.lr)
        loss_fn = nn.BCEWithLogitsLoss()

        pbar = tqdm(range(self.epochs), desc="Training LMF", disable=not self.verbose)
        for epoch in pbar:
            self.train()
            optimizer.zero_grad()

            logits = self(user_idx, item_idx)
            loss = loss_fn(logits, labels)

            loss.backward()
            optimizer.step()

            # Evaluation
            if epoch % 10 == 0 or epoch == self.epochs - 1:
                metrics = {"loss": f"{loss.item():.4f}"}
                if val_data is not None:
                    self.eval()
                    with torch.no_grad():
                        val_logits = self(val_data[0], val_data[1])
                    val_metrics = self._compute_val_metrics(
                        val_logits, val_data[2], loss_fn
                    )
                    metrics.update(val_metrics)
                pbar.set_postfix(metrics)

        self._is_fitted = True
        return self

    def predict_proba(self, X: RollCallMatrix) -> torch.Tensor:
        self._check_is_fitted()
        user_idx, item_idx, _ = X.to_pytorch_tensors()
        user_idx = user_idx.to(self.device)
        item_idx = item_idx.to(self.device)

        self.eval()
        with torch.no_grad():
            logits = self(user_idx, item_idx)
            probs = torch.sigmoid(logits)

        return probs.cpu()

    @property
    def ideal_points_(self):
        """Extract the embeddings interpreting them as ideal points."""
        self._check_is_fitted()
        if self.n_factors > 0:
            return self.user_embedding.weight.detach().cpu().numpy()
        return None


class FMCov(BaseRollCallModel):
    """
    Factorization Machines with Covariates (FM-COV) model.
    Proposed by Maurício Najjar da Silveira (2025).

    Extends Logistic Matrix Factorization by incorporating categorical
    covariates as additive biases and latent dimensional shifts natively.
    """

    def __init__(
        self,
        n_factors: int = 2,
        epochs: int = 1000,
        lr: float = 0.05,
        lambda_alpha: float = 0.0,
        lambda_p: float = 0.0,
        lambda_q: float = 0.0,
        verbose: bool = True,
    ):
        super().__init__()
        self.n_factors = n_factors
        self.epochs = epochs
        self.lr = lr
        self.lambda_alpha = lambda_alpha
        self.lambda_p = lambda_p
        self.lambda_q = lambda_q
        self.verbose = verbose

    def _initialize_parameters(
        self, n_users: int, n_items: int, user_cov_dims: dict, item_cov_dims: dict
    ):
        self._user_cov_dims = user_cov_dims
        self._item_cov_dims = item_cov_dims

        if self.n_factors > 0:
            self.user_embedding = nn.Embedding(n_users, self.n_factors)
            self.item_embedding = nn.Embedding(n_items, self.n_factors)
            nn.init.normal_(self.user_embedding.weight, std=0.1)
            nn.init.normal_(self.item_embedding.weight, std=0.1)

            # Covariate Latent Dim Embeddings: C_{1, i} and C_{2, j}
            self.user_cov_latents = nn.ModuleDict(
                {
                    name: nn.Embedding(dim, self.n_factors)
                    for name, dim in user_cov_dims.items()
                }
            )
            self.item_cov_latents = nn.ModuleDict(
                {
                    name: nn.Embedding(dim, self.n_factors)
                    for name, dim in item_cov_dims.items()
                }
            )
            for emb in self.user_cov_latents.values():
                nn.init.normal_(emb.weight, std=0.1)
            for emb in self.item_cov_latents.values():
                nn.init.normal_(emb.weight, std=0.1)

        self.user_bias = nn.Embedding(n_users, 1)
        self.item_bias = nn.Embedding(n_items, 1)
        self.global_bias = nn.Parameter(torch.zeros(1))

        # Covariate Bias Embeddings: c_{1, i} and c_{2, j}
        self.user_cov_biases = nn.ModuleDict(
            {name: nn.Embedding(dim, 1) for name, dim in user_cov_dims.items()}
        )
        self.item_cov_biases = nn.ModuleDict(
            {name: nn.Embedding(dim, 1) for name, dim in item_cov_dims.items()}
        )

        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)
        for emb in self.user_cov_biases.values():
            nn.init.zeros_(emb.weight)
        for emb in self.item_cov_biases.values():
            nn.init.zeros_(emb.weight)

        self.to(self.device)

    def forward(self, user_idx, item_idx, user_covs, item_covs):
        """
        Compute logit for each user-item pair with covariate side information.

        Args:
            user_idx: Tensor of user (legislator) indices.
            item_idx: Tensor of item (roll call) indices.
            user_covs: Dict mapping covariate name to LongTensor of codes.
            item_covs: Dict mapping covariate name to LongTensor of codes.
        """
        # scalar biases: alpha_p_i = b_p_i + c_1_i
        alpha_p = self.user_bias(user_idx).squeeze(-1)
        for name, emb in self.user_cov_biases.items():
            alpha_p = alpha_p + emb(user_covs[name]).squeeze(-1)

        # scalar biases: alpha_q_j = b_q_j + c_2_j
        alpha_q = self.item_bias(item_idx).squeeze(-1)
        for name, emb in self.item_cov_biases.items():
            alpha_q = alpha_q + emb(item_covs[name]).squeeze(-1)

        if self.n_factors > 0:
            # Latent vectors: P_i = p_i + C_1_i
            P_i = self.user_embedding(user_idx)
            for name, emb in self.user_cov_latents.items():
                P_i = P_i + emb(user_covs[name])

            # Latent vectors: Q_j = q_j + C_2_j
            Q_j = self.item_embedding(item_idx)
            for name, emb in self.item_cov_latents.items():
                Q_j = Q_j + emb(item_covs[name])

            dot = (P_i * Q_j).sum(dim=1)
            return self.global_bias + alpha_p + alpha_q + dot

        return self.global_bias + alpha_p + alpha_q

    def fit(self, X: RollCallMatrix, X_val: RollCallMatrix | None = None) -> "FMCov":
        n_users, n_items = X.shape
        self._initialize_parameters(
            n_users, n_items, X.user_cov_dims, X.item_cov_dims
        )

        user_idx, item_idx, labels = X.to_pytorch_tensors()
        user_idx = user_idx.to(self.device)
        item_idx = item_idx.to(self.device)
        labels = labels.to(self.device)

        # Covariate tensors indexed by observation
        t_user_covs = {
            k: v[user_idx.cpu()].to(self.device)
            for k, v in X.user_cov_tensors.items()
        }
        t_item_covs = {
            k: v[item_idx.cpu()].to(self.device)
            for k, v in X.item_cov_tensors.items()
        }

        val_data = None
        if X_val is not None:
            val_user_idx, val_item_idx, val_labels = X_val.to_pytorch_tensors()
            val_user_idx = val_user_idx.to(self.device)
            val_item_idx = val_item_idx.to(self.device)
            val_labels = val_labels.to(self.device)
            val_t_user_covs = {
                k: v[val_user_idx.cpu()].to(self.device)
                for k, v in X_val.user_cov_tensors.items()
            }
            val_t_item_covs = {
                k: v[val_item_idx.cpu()].to(self.device)
                for k, v in X_val.item_cov_tensors.items()
            }
            val_data = (
                val_user_idx,
                val_item_idx,
                val_labels,
                val_t_user_covs,
                val_t_item_covs,
            )

        optimizer = Adam(self.parameters(), lr=self.lr)
        loss_fn = nn.BCEWithLogitsLoss()

        pbar = tqdm(range(self.epochs), desc="Training FM-COV", disable=not self.verbose)
        for epoch in pbar:
            self.train()
            optimizer.zero_grad()

            logits = self(user_idx, item_idx, t_user_covs, t_item_covs)
            loss = loss_fn(logits, labels)

            # L1 / LASSO regularization to force sparsity across embeddings
            if self.lambda_alpha > 0:
                l1_alpha = torch.norm(self.user_bias.weight, p=1) + torch.norm(
                    self.item_bias.weight, p=1
                )
                for emb in self.user_cov_biases.values():
                    l1_alpha = l1_alpha + torch.norm(emb.weight, p=1)
                for emb in self.item_cov_biases.values():
                    l1_alpha = l1_alpha + torch.norm(emb.weight, p=1)
                loss = loss + self.lambda_alpha * l1_alpha

            if self.n_factors > 0:
                if self.lambda_p > 0:
                    l1_p = torch.norm(self.user_embedding.weight, p=1)
                    for emb in self.user_cov_latents.values():
                        l1_p = l1_p + torch.norm(emb.weight, p=1)
                    loss = loss + self.lambda_p * l1_p
                if self.lambda_q > 0:
                    l1_q = torch.norm(self.item_embedding.weight, p=1)
                    for emb in self.item_cov_latents.values():
                        l1_q = l1_q + torch.norm(emb.weight, p=1)
                    loss = loss + self.lambda_q * l1_q

            loss.backward()
            optimizer.step()

            if epoch % 10 == 0 or epoch == self.epochs - 1:
                metrics = {"loss": f"{loss.item():.4f}"}
                if val_data is not None:
                    self.eval()
                    with torch.no_grad():
                        val_logits = self(
                            val_data[0], val_data[1], val_data[3], val_data[4]
                        )
                    val_metrics = self._compute_val_metrics(
                        val_logits, val_data[2], loss_fn
                    )
                    metrics.update(val_metrics)
                pbar.set_postfix(metrics)

        self._is_fitted = True

        # Compute and cache effective ideal points
        self.eval()
        with torch.no_grad():
            if self.n_factors > 0:
                P_eff = self.user_embedding.weight.clone()
                # Add covariate latent shifts (full legislator-level tensors)
                for name, t in X.user_cov_tensors.items():
                    P_eff = P_eff + self.user_cov_latents[name](t.to(self.device))
                self._cached_ideal_points = P_eff.cpu().numpy()
            else:
                self._cached_ideal_points = None

        return self

    def predict_proba(self, X: RollCallMatrix) -> torch.Tensor:
        self._check_is_fitted()
        user_idx, item_idx, _ = X.to_pytorch_tensors()
        user_idx = user_idx.to(self.device)
        item_idx = item_idx.to(self.device)

        t_user_covs = {
            k: v[user_idx.cpu()].to(self.device)
            for k, v in X.user_cov_tensors.items()
        }
        t_item_covs = {
            k: v[item_idx.cpu()].to(self.device)
            for k, v in X.item_cov_tensors.items()
        }

        self.eval()
        with torch.no_grad():
            logits = self(user_idx, item_idx, t_user_covs, t_item_covs)
            probs = torch.sigmoid(logits)

        return probs.cpu()

    @property
    def ideal_points_(self):
        """
        Extract the effective ideal points (base embedding + covariate shifts).
        Returns a numpy array of shape (n_legislators, n_factors).
        """
        self._check_is_fitted()
        return getattr(self, "_cached_ideal_points", None)
