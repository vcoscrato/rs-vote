import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm.auto import tqdm

from .base import BaseRollCallModel
from ..data.matrix import RollCallMatrix

class LogisticMatrixFactorization(BaseRollCallModel, nn.Module):
    """
    A standard Logistic Matrix Factorization representation.
    Treats Roll Call voting purely as a Collaborative Filtering problem 
    without strict spatial utility assumptions.
    
    Probability of Yea = sigmoid( bias_user + bias_item + dot(embed_user, embed_item) )
    """
    
    def __init__(self, n_factors: int = 2):
        BaseRollCallModel.__init__(self, n_factors=n_factors)
        nn.Module.__init__(self)
        self.n_factors = n_factors
        
    def _initialize_parameters(self, n_users: int, n_items: int):
        self.user_embedding = nn.Embedding(n_users, self.n_factors)
        self.item_embedding = nn.Embedding(n_items, self.n_factors)
        self.user_bias = nn.Embedding(n_users, 1)
        self.item_bias = nn.Embedding(n_items, 1)
        
        # Initialize with small normal distributions
        nn.init.normal_(self.user_embedding.weight, std=0.1)
        nn.init.normal_(self.item_embedding.weight, std=0.1)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)
        
        self.to(self.device)

    def forward(self, user_idx: torch.Tensor, item_idx: torch.Tensor) -> torch.Tensor:
        """
        Calculates the logit (pre-sigmoid) for the given user-item pairs.
        """
        u = self.user_embedding(user_idx)
        i = self.item_embedding(item_idx)
        
        b_u = self.user_bias(user_idx).squeeze()
        b_i = self.item_bias(item_idx).squeeze()
        
        dot = (u * i).sum(dim=1)
        return dot + b_u + b_i

    def fit(self, X: RollCallMatrix, X_val: RollCallMatrix = None, epochs: int = 1000, lr: float = 0.05) -> 'LogisticMatrixFactorization':
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
        
        optimizer = Adam(self.parameters(), lr=lr)
        loss_fn = nn.BCEWithLogitsLoss()
        
        pbar = tqdm(range(epochs), desc="Training LMF")
        for epoch in pbar:
            self.train()
            optimizer.zero_grad()
            
            logits = self(user_idx, item_idx)
            loss = loss_fn(logits, labels)
            
            loss.backward()
            optimizer.step()
            
            # Evaluation
            if X_val is not None and (epoch % 10 == 0 or epoch == epochs - 1):
                self.eval()
                with torch.no_grad():
                    val_logits = self(val_user_idx, val_item_idx)
                    val_loss = loss_fn(val_logits, val_labels)
                    
                    val_preds = (torch.sigmoid(val_logits) > 0.5).float()
                    val_acc = val_preds.eq(val_labels).float().mean()
                    
                    pbar.set_postfix({'loss': f"{loss.item():.4f}", 'val_loss': f"{val_loss.item():.4f}", 'val_acc': f"{val_acc.item():.4f}"})
            elif epoch % 10 == 0 or epoch == epochs - 1:
                pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
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
        return self.user_embedding.weight.detach().cpu().numpy()
