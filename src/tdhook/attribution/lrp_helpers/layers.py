"""
LRP-compatible layers.

This code is adapted from the Zennit library (LGPL-3.0)
Original source: https://github.com/chr5tphr/zennit/blob/main/src/zennit/layer.py
"""

import torch


class Sum(torch.nn.Module):
    """Compute the sum along an axis.

    Parameters
    ----------
    dim : int
        Dimension over which to sum.
    """

    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, input):
        """Computes the sum along a dimension.

        Parameters
        ----------
        input: :py:obj:`torch.Tensor`
            The input on which to sum.

        Returns
        -------
        :py:obj:`torch.Tensor`
            The resulting tensor summed along dimension `dim`.

        """
        return torch.sum(input, dim=self.dim)
