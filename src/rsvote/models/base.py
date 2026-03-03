import inspect
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from sklearn.metrics import f1_score, roc_auc_score

from ..data.matrix import RollCallMatrix


class BaseRollCallModel(nn.Module, ABC):
    """
    Abstract Base Class for Roll Call scaling methods.
    Mimics a scikit-learn estimator interface but optimized with PyTorch.
    """

    def __init__(self):
        super().__init__()
        self._is_fitted = False
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __repr__(self):
        # Introspect __init__ signature to display hyperparameters
        sig = inspect.signature(self.__init__)
        params = []
        for name, param in sig.parameters.items():
            if name == "self":
                continue
            val = getattr(self, name, param.default)
            if val != param.default:
                params.append(f"{name}={val!r}")
        return f"{self.__class__.__name__}({', '.join(params)})"

    @abstractmethod
    def fit(
        self,
        X: RollCallMatrix,
        X_val: RollCallMatrix | None = None,
    ) -> "BaseRollCallModel":
        """
        Fit the model to the roll call data.

        Args:
            X: Training roll call matrix.
            X_val: Optional validation roll call matrix for monitoring.
        """
        pass

    @abstractmethod
    def predict_proba(self, X: RollCallMatrix) -> torch.Tensor:
        """
        Predict the likelihood of a Yea vote for specific user-item pairs.

        Args:
            X: RollCallMatrix containing the pairs to predict.

        Returns:
            A tensor of probabilities [0, 1].
        """
        pass

    def predict(self, X: RollCallMatrix) -> torch.Tensor:
        """
        Predict binary vote outcomes (0 or 1).
        """
        probs = self.predict_proba(X)
        return (probs > 0.5).float()

    def _check_is_fitted(self):
        if not self._is_fitted:
            raise RuntimeError(
                f"This {self.__class__.__name__} instance is not fitted yet. "
                "Call 'fit' with appropriate arguments before using this estimator."
            )

    @staticmethod
    def _compute_val_metrics(
        val_logits: torch.Tensor, val_labels: torch.Tensor, loss_fn
    ) -> dict:
        """
        Compute standard validation metrics from logits and labels.

        Returns a dict with v_loss, v_acc, v_auc, v_f1 as formatted strings.
        """
        with torch.no_grad():
            val_loss = loss_fn(val_logits, val_labels)

            val_probs = torch.sigmoid(val_logits)
            val_preds = (val_probs > 0.5).float()
            val_acc = val_preds.eq(val_labels).float().mean()

            try:
                # Move to CPU for sklearn metrics
                labels_np = val_labels.cpu().numpy()
                probs_np = val_probs.cpu().numpy()
                preds_np = val_preds.cpu().numpy()

                val_auc = roc_auc_score(labels_np, probs_np)
                val_f1 = f1_score(labels_np, preds_np)
            except ValueError:
                val_auc = float("nan")
                val_f1 = float("nan")

        return {
            "v_loss": f"{val_loss.item():.4f}",
            "v_acc": f"{val_acc.item():.4f}",
            "v_auc": f"{val_auc:.4f}",
            "v_f1": f"{val_f1:.4f}",
        }
