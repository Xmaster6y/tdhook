"""
GPT2 cache task.
"""

import torch

impact_parameters = {
    "model_size": ["gpt2", "gpt2-medium", "gpt2-large"],
    "batch_size": [10, 50, 100, 500],
    "variation": ["all"],
}
default_parameters = {
    "model_size": "gpt2",
    "batch_size": 50,
    "variation": "all",
}


def assert_accuracy(outs1, outs2):
    """Assert accuracy of two outputs."""
    for out1, out2 in zip(outs1[0], outs2[0]):
        assert torch.allclose(out1, out2)
