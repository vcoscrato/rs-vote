import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm.auto import tqdm

from .base import BaseRollCallModel
from ..data.matrix import RollCallMatrix

class WNominate(BaseRollCallModel, nn.Module):
    """
    PyTorch implementation of the W-NOMINATE spatial model.
    
    Instead of Alternating Optimization, we use Gradient Descent to minimize 
    the Negative Log-Likelihood of the Vote Matrix.
    
    Utility(yea) = exp(-0.5 * ||x_i - z_jy||^2)
    Probability(yea) = Utility(yea) / (Utility(yea) + Utility(nay))
                     = sigmoid( 0.5 * (||x_i - z_jn||^2 - ||x_i - z_jy||^2) )
                     
    We reparameterize the votes as a midpoint `z_mid` and a direction `z_spread` 
    to simplify optimization, which is standard in more recent implementations.
    """
    
    def __init__(self, n_dims: int = 2, beta: float = 15.0):
        BaseRollCallModel.__init__(self, n_dims=n_dims, beta=beta)
        nn.Module.__init__(self)
        self.n_dims = n_dims
        self.beta = beta # Temperature parameter
        
    def _initialize_parameters(self, n_users: int, n_items: int):
        self.ideal_points = nn.Embedding(n_users, self.n_dims)
        self.vote_midpoints = nn.Embedding(n_items, self.n_dims)
        self.vote_spreads = nn.Embedding(n_items, self.n_dims)
        
        # W-NOMINATE bounds ideal points between -1 and 1
        nn.init.uniform_(self.ideal_points.weight, -0.5, 0.5)
        nn.init.uniform_(self.vote_midpoints.weight, -0.5, 0.5)
        nn.init.normal_(self.vote_spreads.weight, std=0.5)
        
        self.to(self.device)

    def forward(self, user_idx: torch.Tensor, item_idx: torch.Tensor) -> torch.Tensor:
        """
        Calculates the logit (pre-sigmoid) difference in utilities.
        """
        x_i = self.ideal_points(user_idx)
        z_mid = self.vote_midpoints(item_idx)
        z_spread = self.vote_spreads(item_idx)
        
        # The logit in this parameterization is proportional to the 
        # dot product of the distance from the midpoint and the spread vector.
        # Logit = beta * (x_i - z_mid) . z_spread
        
        diff = x_i - z_mid
        logit = self.beta * (diff * z_spread).sum(dim=1)
        
        return logit

    def fit(self, X: RollCallMatrix, X_val: RollCallMatrix = None, epochs: int = 1500, lr: float = 0.05) -> 'WNominate':
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
        
        pbar = tqdm(range(epochs), desc="Training W-NOMINATE")
        for epoch in pbar:
            self.train()
            optimizer.zero_grad()
            
            logits = self(user_idx, item_idx)
            loss = loss_fn(logits, labels)
            
            # L2 Regularization (weight decay equivalent) to constrain space
            l2_reg = torch.tensor(0., device=self.device)
            for param in self.parameters():
                l2_reg += torch.norm(param)
            loss += 0.001 * l2_reg
            
            loss.backward()
            optimizer.step()
            
            # W-Nominate traditionally constrains ideal points to the unit hypersphere
            with torch.no_grad():
                norms = torch.norm(self.ideal_points.weight, dim=1, keepdim=True)
                mask = norms > 1.0
                if mask.any():
                    self.ideal_points.weight[mask.squeeze()] /= norms[mask.squeeze()]
            
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
        self._check_is_fitted()
        return self.ideal_points.weight.detach().cpu().numpy()
