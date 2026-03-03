import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm.auto import tqdm

from ..data.matrix import RollCallMatrix
from .base import BaseRollCallModel


class WNominate(BaseRollCallModel):
    """
    PyTorch implementation of the W-NOMINATE spatial voting model.

    Follows the original Poole & Rosenthal formulation exactly:

        Utility(yea) = beta * exp(-0.5 * sum_k[ w_k^2 * (x_ik - z_jy_k)^2 ])
        Utility(nay) = beta * exp(-0.5 * sum_k[ w_k^2 * (x_ik - z_jn_k)^2 ])

        Probability(yea) = sigmoid( Utility(yea) - Utility(nay) )

    Where x_i is the legislator ideal point, z_jy is the yea outcome location,
    z_jn is the nay outcome location, beta is the signal-to-noise parameter,
    and w_k are dimensional weights.

    Using the midpoint/spread reparameterization:
        z_jy = z_mid_j + z_spread_j
        z_jn = z_mid_j - z_spread_j

    Parameters:
        - beta: Signal-to-noise ratio (estimated, initialized to 15.0)
        - w_1 = 1.0: First dimension weight (fixed for identification)
        - w_k (k>=2): Estimated dimension weights (initialized to 0.5)
        - Ideal points are constrained to the unit hypersphere (||x_i|| <= 1)

    Two optimization methods are supported:
        - "gradient_descent": Joint SGD over all parameters
        - "alternating": Three-step procedure matching the original algorithm
    """

    def __init__(
        self,
        n_dims: int = 2,
        init_beta: float = 15.0,
        init_weights: float = 0.5,
        method: str = "gradient_descent",
        epochs: int = 1500,
        lr: float = 0.05,
        alternating_inner_steps: int = 50,
        convergence_corr: float = 0.99,
        verbose: bool = True,
    ):
        super().__init__()
        if n_dims < 1:
            raise ValueError(
                "WNominate requires at least 1 spatial dimension (n_dims >= 1)."
            )
        if method not in ["gradient_descent", "alternating"]:
            raise ValueError("method must be 'gradient_descent' or 'alternating'")

        self.n_dims = n_dims
        self.init_beta = init_beta
        self.init_weights = init_weights
        self.method = method
        self.epochs = epochs
        self.lr = lr
        self.alternating_inner_steps = alternating_inner_steps
        self.convergence_corr = convergence_corr
        self.verbose = verbose

    def _initialize_parameters(self, n_users: int, n_items: int):
        # Learnable beta (signal-to-noise)
        self.beta = nn.Parameter(torch.tensor(self.init_beta))

        # Dimension weights: w_1 = 1.0 (fixed), w_k (k>=2) learnable
        if self.n_dims > 1:
            self.dim_weights = nn.Parameter(
                torch.full((self.n_dims - 1,), self.init_weights)
            )

        # Spatial parameters
        self.ideal_points = nn.Embedding(n_users, self.n_dims)
        self.vote_midpoints = nn.Embedding(n_items, self.n_dims)
        self.vote_spreads = nn.Embedding(n_items, self.n_dims)

        nn.init.uniform_(self.ideal_points.weight, -0.5, 0.5)
        nn.init.uniform_(self.vote_midpoints.weight, -0.5, 0.5)
        nn.init.normal_(self.vote_spreads.weight, std=0.5)

        self.to(self.device)

    def _get_weights_squared(self) -> torch.Tensor:
        w1 = torch.ones(1, device=self.device)
        if self.n_dims > 1:
            return torch.cat([w1, self.dim_weights**2])
        return w1

    def forward(self, user_idx: torch.Tensor, item_idx: torch.Tensor) -> torch.Tensor:
        """
        Computes the logit (pre-sigmoid) for the given user-item pairs using
        the exact Gaussian utility difference.
        """
        x_i = self.ideal_points(user_idx)  # (N, D)
        z_mid = self.vote_midpoints(item_idx)  # (N, D)
        z_spread = self.vote_spreads(item_idx)  # (N, D)

        z_yea = z_mid + z_spread
        z_nay = z_mid - z_spread

        w_sq = self._get_weights_squared()  # (D,)

        # Distances
        d_yea_sq = (w_sq * (x_i - z_yea) ** 2).sum(dim=1)
        d_nay_sq = (w_sq * (x_i - z_nay) ** 2).sum(dim=1)

        # Utilities
        u_yea = self.beta * torch.exp(-0.5 * d_yea_sq)
        u_nay = self.beta * torch.exp(-0.5 * d_nay_sq)

        return u_yea - u_nay

    def _project_ideal_points(self):
        """Project ideal points back onto the unit ball (||x_i|| <= 1)."""
        with torch.no_grad():
            norms = torch.norm(self.ideal_points.weight, dim=1, keepdim=True)
            mask = (norms > 1.0).squeeze(-1)
            if mask.any():
                self.ideal_points.weight[mask] /= norms[mask]

    def fit(self, X: RollCallMatrix, X_val: RollCallMatrix | None = None) -> "WNominate":
        n_users, n_items = X.shape
        self._initialize_parameters(n_users, n_items)

        user_idx, item_idx, labels = X.to_pytorch_tensors()
        user_idx = user_idx.to(self.device)
        item_idx = item_idx.to(self.device)
        labels = labels.to(self.device)

        val_data = None
        if X_val is not None:
            v_u, v_i, v_l = X_val.to_pytorch_tensors()
            val_data = (v_u.to(self.device), v_i.to(self.device), v_l.to(self.device))

        loss_fn = nn.BCEWithLogitsLoss()

        if self.method == "alternating":
            self._fit_alternating(user_idx, item_idx, labels, val_data, loss_fn)
        else:
            self._fit_gradient_descent(user_idx, item_idx, labels, val_data, loss_fn)

        self._is_fitted = True
        return self

    def _fit_gradient_descent(self, user_idx, item_idx, labels, val_data, loss_fn):
        optimizer = Adam(self.parameters(), lr=self.lr)

        pbar = tqdm(
            range(self.epochs), desc="W-NOMINATE (SGD)", disable=not self.verbose
        )
        for epoch in pbar:
            self.train()
            optimizer.zero_grad()

            logits = self(user_idx, item_idx)
            loss = loss_fn(logits, labels)

            loss.backward()
            optimizer.step()
            self._project_ideal_points()

            if epoch % 10 == 0 or epoch == self.epochs - 1:
                metrics = {"loss": f"{loss.item():.4f}", "β": f"{self.beta.item():.2f}"}
                if self.n_dims > 1:
                    w_str = ",".join(
                        f"{w:.2f}" for w in self.dim_weights.detach().cpu().tolist()
                    )
                    metrics["w"] = w_str
                if val_data is not None:
                    self.eval()
                    with torch.no_grad():
                        val_logits = self(val_data[0], val_data[1])
                    val_metrics = self._compute_val_metrics(
                        val_logits, val_data[2], loss_fn
                    )
                    metrics.update(val_metrics)
                pbar.set_postfix(metrics)

    def _fit_alternating(self, user_idx, item_idx, labels, val_data, loss_fn):
        bill_params = [self.vote_midpoints.weight, self.vote_spreads.weight]
        leg_params = [self.ideal_points.weight]
        hyper_params = [self.beta]
        if self.n_dims > 1:
            hyper_params.append(self.dim_weights)

        opt_bills = Adam(bill_params, lr=self.lr)
        opt_legs = Adam(leg_params, lr=self.lr)
        opt_hyper = Adam(hyper_params, lr=self.lr * 0.1)

        prev_all_params = None

        pbar = tqdm(
            range(self.epochs),
            desc="W-NOMINATE (Alternating)",
            disable=not self.verbose,
        )
        for outer in pbar:
            # Step 1: Optimize bill parameters
            self._freeze_all_except(bill_params)
            for _ in range(self.alternating_inner_steps):
                opt_bills.zero_grad()
                logits = self(user_idx, item_idx)
                loss = loss_fn(logits, labels)
                loss.backward()
                opt_bills.step()

            # Step 2: Optimize legislator ideal points
            self._freeze_all_except(leg_params)
            for _ in range(self.alternating_inner_steps):
                opt_legs.zero_grad()
                logits = self(user_idx, item_idx)
                loss = loss_fn(logits, labels)
                loss.backward()
                opt_legs.step()
                self._project_ideal_points()

            # Step 3: Optimize beta and dimension weights
            self._freeze_all_except(hyper_params)
            for _ in range(self.alternating_inner_steps):
                opt_hyper.zero_grad()
                logits = self(user_idx, item_idx)
                loss = loss_fn(logits, labels)
                loss.backward()
                opt_hyper.step()

            for p in self.parameters():
                p.requires_grad = True

            with torch.no_grad():
                new_all_params = torch.cat([p.data.flatten() for p in self.parameters()])

            converged = False
            if prev_all_params is not None:
                corr = torch.corrcoef(torch.stack([prev_all_params, new_all_params]))[
                    0, 1
                ]
                converged = corr.item() >= self.convergence_corr

            prev_all_params = new_all_params

            metrics = {"loss": f"{loss.item():.4f}", "β": f"{self.beta.item():.2f}"}
            if self.n_dims > 1:
                w_str = ",".join(
                    f"{w:.2f}" for w in self.dim_weights.detach().cpu().tolist()
                )
                metrics["w"] = w_str
            if val_data is not None:
                self.eval()
                with torch.no_grad():
                    val_logits = self(val_data[0], val_data[1])
                val_metrics = self._compute_val_metrics(val_logits, val_data[2], loss_fn)
                metrics.update(val_metrics)
            pbar.set_postfix(metrics)

            if converged:
                if self.verbose:
                    pbar.write(
                        f"Converged at outer iteration {outer} (corr >= {self.convergence_corr})"
                    )
                break

    def _freeze_all_except(self, active_params: list):
        active_set = {id(p) for p in active_params}
        for p in self.parameters():
            p.requires_grad = id(p) in active_set

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
        self._check_is_fitted()
        return self.ideal_points.weight.detach().cpu().numpy()
