"""
Estimators for probing.
"""

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


class MeanDifferenceClassifier:
    def __init__(self, normalize: bool = True):
        self._normalize = normalize
        self._coef = None
        self._intercept = None

    @property
    def coef_(self):
        if self._coef is None:
            raise ValueError("Model not fitted")
        return self._coef

    @property
    def intercept_(self):
        if self._intercept is None:
            raise ValueError("Model not fitted")
        return self._intercept

    def fit(self, X, y):
        if len(y.shape) > 1:
            raise ValueError("Multiclass classification not supported")
        y = np.expand_dims(y, 1)
        pos_count = y.sum()
        neg_count = (1 - y).sum()
        if pos_count == 0 or neg_count == 0:
            raise ValueError("Both classes must be present in y")
        pos = (X * y).sum(axis=0) / pos_count
        neg = (X * (1 - y)).sum(axis=0) / neg_count
        pos_norm = np.linalg.norm(pos)
        neg_norm = np.linalg.norm(neg)

        self._coef = pos - neg
        self._intercept = -0.5 * (pos_norm**2 - neg_norm**2)
        if self._normalize:
            coef_norm = np.linalg.norm(self._coef)
            if coef_norm > 0:
                self._intercept = self._intercept / coef_norm
                self._coef = self._coef / coef_norm

        self._intercept = self._intercept.reshape((1,))
        self._coef = self._coef.reshape((1, -1))

    def _decision_function(self, X):
        return (X * self._coef).sum(axis=1) + self._intercept

    def predict(self, X):
        return self._decision_function(X) > 0

    def predict_proba(self, X):
        pos_proba = 1 / (1 + np.exp(-self._decision_function(X)))
        neg_proba = 1 - pos_proba
        return np.stack([neg_proba, pos_proba], axis=1)


class TorchEstimator(nn.Module):
    """Base class for torch estimators."""

    def __init__(
        self,
        *,
        num_classes: Optional[int] = None,
        epochs: int = 100,
        lr: float = 1e-3,
        batch_size: int = 128,
        device: Optional[torch.device] = None,
        verbose: bool = False,
    ):
        super().__init__()
        self._num_classes = num_classes
        self._epochs = epochs
        self._lr = lr
        self._batch_size = batch_size
        self._device = device
        self._verbose = verbose

    def fit(self, *Xs: torch.Tensor, y: torch.Tensor):
        dataset = TensorDataset(*Xs, y)
        train_loader = DataLoader(dataset, batch_size=self._batch_size, shuffle=True)
        optimizer = torch.optim.Adam(self.parameters(), lr=self._lr)
        for epoch in range(self._epochs):
            self.train()
            for batch in train_loader:
                *Xs_b, y_b = batch
                o_b = self(*Xs_b)
                loss = self._loss_fn(o_b, y_b)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            if self._verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{self._epochs}")
        return self

    def predict(self, *Xs: torch.Tensor) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            Y = self(*Xs)
            return Y.argmax(dim=-1) if self._num_classes is not None else Y

    def _loss_fn(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self._num_classes is None:  # regression
            if output.shape != target.shape:
                raise ValueError(f"Output shape {output.shape} does not match target shape {target.shape}")
            return F.mse_loss(output, target)
        else:
            if output.shape[:-1] != target.shape or output.shape[-1] != self._num_classes:
                raise ValueError(f"Output shape {output.shape} does not match target shape {target.shape}")
            return F.cross_entropy(output, target)


class LinearEstimator(TorchEstimator):
    """Linear estimator: W h + b."""

    def __init__(
        self,
        d_latent: int,
        bias: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.linear = nn.Linear(d_latent, self._num_classes or 1, bias=bias, device=self._device)

    def forward(self, *Xs: torch.Tensor) -> torch.Tensor:
        if len(Xs) != 1:
            raise ValueError(f"Linear estimator expects 1 input tensor, got {len(Xs)}")
        return self.linear(Xs[0])


class BilinearEstimator(TorchEstimator):
    """Bilinear estimator: h_1^T A h_2 + b."""

    def __init__(
        self,
        d_latent1: int,
        d_latent2: int,
        bias: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.bilinear = nn.Bilinear(d_latent1, d_latent2, self._num_classes or 1, bias=bias, device=self._device)

    def forward(self, h1: torch.Tensor, h2: torch.Tensor) -> torch.Tensor:
        return self.bilinear(h1, h2)


class LowRankBilinearEstimator(TorchEstimator):
    """Low-rank bilinear: (U h_1) * (V h_2) + b."""

    def __init__(
        self,
        d_latent1: int,
        d_latent2: int,
        bias: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        out_features = self._num_classes or 1
        self.linear1 = nn.Linear(d_latent1, out_features, device=self._device)
        self.linear2 = nn.Linear(d_latent2, out_features, device=self._device)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, device=self._device))
        else:
            self.register_parameter("bias", None)

    def forward(self, h1: torch.Tensor, h2: torch.Tensor) -> torch.Tensor:
        output = self.linear1(h1) * self.linear2(h2)
        if self.bias is not None:
            output = output + self.bias
        return output
