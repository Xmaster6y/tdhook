"""
LRP task.
"""

import torch


impact_parameters = {
    "width": [100, 1000, 5_000, 10_000],
    "height": [5, 10, 20, 50],
    "batch_size": [100, 1000, 10_000, 100_000],
    "variation": ["epsilon", "alpha-beta"],
}
default_parameters = {
    "width": 1000,
    "height": 10,
    "batch_size": 1000,
    "variation": "epsilon",
}


def assert_accuracy(out1, out2):
    """Assert accuracy of two outputs."""
    assert torch.allclose(out1, out2)
