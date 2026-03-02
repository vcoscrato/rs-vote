from abc import ABC, abstractmethod

import torch

from ..data.matrix import RollCallMatrix


class BaseRollCallModel(ABC):
    """
    Abstract Base Class for Roll Call scaling methods.
    Mimics a scikit-learn estimator interface but optimized with PyTorch.
    """

    def __init__(self, **kwargs):
        self.params = kwargs
        self._is_fitted = False
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @abstractmethod
    def fit(
        self,
        X: RollCallMatrix,
        X_val: RollCallMatrix | None = None,
        epochs: int = 1000,
        lr: float = 0.01,
        verbose: bool = True,
    ) -> BaseRollCallModel:
        """
        Fit the model to the roll call data.
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

    def _check_is_fitted(self):
        if not self._is_fitted:
            raise RuntimeError(
                "This instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator."
            )
