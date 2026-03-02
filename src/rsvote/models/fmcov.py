import torch
import torch.nn as nn
from sklearn.metrics import f1_score, roc_auc_score
from torch.optim import Adam
from tqdm.auto import tqdm

from ..data.matrix import RollCallMatrix
from .base import BaseRollCallModel


class FMCov(BaseRollCallModel, nn.Module):
    """
    Factorization Machines with Covariates (FM-COV) model.
    Proposed by Maurício Najjar da Silveira (2025).

    Extends Logistic Matrix Factorization by incorporating categorical
    covariates as additive biases and latent dimensional shifts natively.
    """

    def __init__(self, n_factors: int = 2):
        BaseRollCallModel.__init__(self, n_factors=n_factors)
        nn.Module.__init__(self)
        self.n_factors = n_factors
        # Save covariates config for prediction
        self.user_cov_dims = {}
        self.item_cov_dims = {}

    def _initialize_parameters(
        self, n_users: int, n_items: int, user_cov_dims: dict, item_cov_dims: dict
    ):
        self.user_cov_dims = user_cov_dims
        self.item_cov_dims = item_cov_dims

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
        # scalar biases: alpha_p_i = b_p_i + c_1_i
        alpha_p = self.user_bias(user_idx).squeeze(-1)
        for name, emb in self.user_cov_biases.items():
            alpha_p += emb(user_covs[name]).squeeze(-1)

        # scalar biases: alpha_q_j = b_q_j + c_2_j
        alpha_q = self.item_bias(item_idx).squeeze(-1)
        for name, emb in self.item_cov_biases.items():
            alpha_q += emb(item_covs[name]).squeeze(-1)

        if self.n_factors > 0:
            # Latent vectors: P_i = p_i + C_1_i
            P_i = self.user_embedding(user_idx)
            for name, emb in self.user_cov_latents.items():
                P_i += emb(user_covs[name])

            # Latent vectors: Q_j = q_j + C_2_j
            Q_j = self.item_embedding(item_idx)
            for name, emb in self.item_cov_latents.items():
                Q_j += emb(item_covs[name])

            dot = (P_i * Q_j).sum(dim=1)
            return self.global_bias + alpha_p + alpha_q + dot

        return self.global_bias + alpha_p + alpha_q

    def fit(
        self,
        X: RollCallMatrix,
        X_val: RollCallMatrix | None = None,
        epochs: int = 1000,
        lr: float = 0.05,
        verbose: bool = True,
        lambda_alpha: float = 0.0,
        lambda_p: float = 0.0,
        lambda_q: float = 0.0,
    ) -> FMCov:
        n_users, n_items = X.shape
        user_covs_np, user_dims = X.get_user_covariates()
        item_covs_np, item_dims = X.get_item_covariates()

        self._initialize_parameters(n_users, n_items, user_dims, item_dims)

        user_idx, item_idx, labels = X.to_pytorch_tensors()
        user_idx = user_idx.to(self.device)
        item_idx = item_idx.to(self.device)
        labels = labels.to(self.device)

        # Build tensor mappings for covariates
        t_user_covs = {
            k: torch.tensor(
                v[user_idx.cpu().numpy()], dtype=torch.long, device=self.device
            )
            for k, v in user_covs_np.items()
        }
        t_item_covs = {
            k: torch.tensor(
                v[item_idx.cpu().numpy()], dtype=torch.long, device=self.device
            )
            for k, v in item_covs_np.items()
        }

        if X_val is not None:
            val_user_idx, val_item_idx, val_labels = X_val.to_pytorch_tensors()
            val_user_idx = val_user_idx.to(self.device)
            val_item_idx = val_item_idx.to(self.device)
            val_labels = val_labels.to(self.device)

            val_user_covs_np, _ = X_val.get_user_covariates()
            val_item_covs_np, _ = X_val.get_item_covariates()
            val_t_user_covs = {
                k: torch.tensor(
                    v[val_user_idx.cpu().numpy()], dtype=torch.long, device=self.device
                )
                for k, v in val_user_covs_np.items()
            }
            val_t_item_covs = {
                k: torch.tensor(
                    v[val_item_idx.cpu().numpy()], dtype=torch.long, device=self.device
                )
                for k, v in val_item_covs_np.items()
            }

        optimizer = Adam(self.parameters(), lr=lr)
        loss_fn = nn.BCEWithLogitsLoss()

        pbar = tqdm(range(epochs), desc="Training FM-COV", disable=not verbose)
        for epoch in pbar:
            self.train()
            optimizer.zero_grad()

            logits = self(user_idx, item_idx, t_user_covs, t_item_covs)
            loss = loss_fn(logits, labels)

            # L1 / LASSO regularization to force sparsity across embeddings explicitly
            if lambda_alpha > 0:
                l1_alpha = torch.norm(self.user_bias.weight, p=1) + torch.norm(
                    self.item_bias.weight, p=1
                )
                for emb in self.user_cov_biases.values():
                    l1_alpha = l1_alpha + torch.norm(emb.weight, p=1)
                for emb in self.item_cov_biases.values():
                    l1_alpha = l1_alpha + torch.norm(emb.weight, p=1)
                loss = loss + lambda_alpha * l1_alpha

            if self.n_factors > 0:
                if lambda_p > 0:
                    l1_p = torch.norm(self.user_embedding.weight, p=1)
                    for emb in self.user_cov_latents.values():
                        l1_p = l1_p + torch.norm(emb.weight, p=1)
                    loss = loss + lambda_p * l1_p
                if lambda_q > 0:
                    l1_q = torch.norm(self.item_embedding.weight, p=1)
                    for emb in self.item_cov_latents.values():
                        l1_q = l1_q + torch.norm(emb.weight, p=1)
                    loss = loss + lambda_q * l1_q

            loss.backward()
            optimizer.step()

            if X_val is not None and (epoch % 10 == 0 or epoch == epochs - 1):
                self.eval()
                with torch.no_grad():
                    val_logits = self(
                        val_user_idx, val_item_idx, val_t_user_covs, val_t_item_covs
                    )
                    val_loss = loss_fn(val_logits, val_labels)

                    val_probs = torch.sigmoid(val_logits)
                    val_preds = (val_probs > 0.5).float()
                    val_acc = val_preds.eq(val_labels).float().mean()

                    try:
                        val_auc = roc_auc_score(
                            val_labels.cpu().numpy(), val_probs.cpu().numpy()
                        )
                        val_f1 = f1_score(
                            val_labels.cpu().numpy(), val_preds.cpu().numpy()
                        )
                    except ValueError:
                        val_auc = float("nan")
                        val_f1 = float("nan")

                    pbar.set_postfix(
                        {
                            "loss": f"{loss.item():.4f}",
                            "v_loss": f"{val_loss.item():.4f}",
                            "v_acc": f"{val_acc.item():.4f}",
                            "v_auc": f"{val_auc:.4f}",
                            "v_f1": f"{val_f1:.4f}",
                        }
                    )
            elif epoch % 10 == 0 or epoch == epochs - 1:
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        self._is_fitted = True
        return self

    def predict_proba(self, X: RollCallMatrix) -> torch.Tensor:
        self._check_is_fitted()
        user_idx, item_idx, _ = X.to_pytorch_tensors()
        user_idx = user_idx.to(self.device)
        item_idx = item_idx.to(self.device)

        user_covs_np, _ = X.get_user_covariates()
        item_covs_np, _ = X.get_item_covariates()

        t_user_covs = {
            k: torch.tensor(
                v[user_idx.cpu().numpy()], dtype=torch.long, device=self.device
            )
            for k, v in user_covs_np.items()
        }
        t_item_covs = {
            k: torch.tensor(
                v[item_idx.cpu().numpy()], dtype=torch.long, device=self.device
            )
            for k, v in item_covs_np.items()
        }

        self.eval()
        with torch.no_grad():
            logits = self(user_idx, item_idx, t_user_covs, t_item_covs)
            probs = torch.sigmoid(logits)

        return probs.cpu()

    @property
    def ideal_points_(self):
        self._check_is_fitted()
        if self.n_factors > 0:
            return self.user_embedding.weight.detach().cpu().numpy()
        return None
