import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm.auto import tqdm

from ..data.matrix import RollCallMatrix
from .wnominate import WNominate


class AlternatingWNominate(WNominate):
    """
    Demonstrates the flexibility of the API to handle Alternating Optimization
    methods (similar to Alternating Least Squares or the original W-NOMINATE
    Alternating Maximum Likelihood approach).

    Instead of allowing the gradients to flow through all parameters simultaneously,
    this implementation isolates the optimization into blocks:
    - Step 1: Hold vote parameters constant, optimize legislator ideal points.
    - Step 2: Hold legislator ideal points constant, optimize vote parameters.
    """

    def fit(
        self,
        X: RollCallMatrix,
        X_val: RollCallMatrix = None,
        epochs: int = 1500,
        lr: float = 0.05,
    ) -> AlternatingWNominate:
        n_users, n_items = X.shape
        self._initialize_parameters(n_users, n_items)

        user_idx, item_idx, labels = X.to_pytorch_tensors()
        user_idx = user_idx.to(self.device)
        item_idx = item_idx.to(self.device)
        labels = labels.to(self.device)

        if X_val is not None:
            val_user_idx, val_item_idx, val_labels = X_val.to_pytorch_tensors()
            val_user_idx = val_user_idx.to(self.device)
            val_item_idx = val_item_idx.to(self.device)
            val_labels = val_labels.to(self.device)

        # In Alternating Optimization, we segregate the parameters into individual optimizers
        opt_legislators = Adam([self.ideal_points.weight], lr=lr)
        opt_votes = Adam([self.vote_midpoints.weight, self.vote_spreads.weight], lr=lr)

        loss_fn = nn.BCEWithLogitsLoss()

        pbar = tqdm(range(epochs), desc="Training Alternating W-NOMINATE")
        for epoch in pbar:
            self.train()

            # Alternate the frozen parameters every 10 epochs
            # True ALS would run each to convergence, but block-interleaved
            # gradient descent is often more stable in Deep Learning frameworks.
            optimize_legislators = (epoch // 10) % 2 == 0

            if optimize_legislators:
                active_optimizer = opt_legislators
                # Freeze vote params
                self.vote_midpoints.weight.requires_grad = False
                self.vote_spreads.weight.requires_grad = False
                self.ideal_points.weight.requires_grad = True
            else:
                active_optimizer = opt_votes
                # Freeze legislator params
                self.ideal_points.weight.requires_grad = False
                self.vote_midpoints.weight.requires_grad = True
                self.vote_spreads.weight.requires_grad = True

            active_optimizer.zero_grad()

            logits = self(user_idx, item_idx)
            loss = loss_fn(logits, labels)

            # L2 Regularization
            l2_reg = torch.tensor(0.0, device=self.device)
            # Only penalize active parameters
            if optimize_legislators:
                l2_reg += torch.norm(self.ideal_points.weight)
            else:
                l2_reg += torch.norm(self.vote_midpoints.weight)
                l2_reg += torch.norm(self.vote_spreads.weight)

            loss += 0.001 * l2_reg

            loss.backward()
            active_optimizer.step()

            # Enforce unit hypersphere constraint on ideal points
            if optimize_legislators:
                with torch.no_grad():
                    norms = torch.norm(self.ideal_points.weight, dim=1, keepdim=True)
                    mask = norms > 1.0
                    if mask.any():
                        self.ideal_points.weight[mask.squeeze()] /= norms[
                            mask.squeeze()
                        ]

            # Evaluation
            if X_val is not None and (epoch % 10 == 0 or epoch == epochs - 1):
                self.eval()
                with torch.no_grad():
                    val_logits = self(val_user_idx, val_item_idx)
                    val_loss = loss_fn(val_logits, val_labels)

                    val_preds = (torch.sigmoid(val_logits) > 0.5).float()
                    val_acc = val_preds.eq(val_labels).float().mean()

                    pbar.set_postfix(
                        {
                            "loss": f"{loss.item():.4f}",
                            "val_loss": f"{val_loss.item():.4f}",
                            "val_acc": f"{val_acc.item():.4f}",
                            "block": "Legs" if optimize_legislators else "Votes",
                        }
                    )
            elif epoch % 10 == 0 or epoch == epochs - 1:
                pbar.set_postfix(
                    {
                        "loss": f"{loss.item():.4f}",
                        "block": "Legs" if optimize_legislators else "Votes",
                    }
                )

        self._is_fitted = True
        return self
