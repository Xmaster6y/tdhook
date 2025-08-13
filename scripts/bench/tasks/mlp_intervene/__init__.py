"""
MLP intervention task.
"""

import torch


impact_parameters = {
    "width": [100, 1000, 5_000, 10_000],
    "height": [5, 10, 20, 50],
    "batch_size": [100, 1000, 10_000, 100_000],
    "variation": [5, 10, 20, 50],
}
default_parameters = {
    "width": 1000,
    "height": 10,
    "batch_size": 1000,
    "variation": 5,
}


def assert_accuracy(outs1, outs2):
    """Assert accuracy of two outputs."""
    for out1, out2 in zip(outs1, outs2):
        assert torch.allclose(out1, out2)
