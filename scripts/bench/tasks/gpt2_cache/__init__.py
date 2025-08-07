"""
GPT2 cache task.
"""

import torch

impact_parameters = {
    "model_size": ["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"],
    "batch_size": [100, 1000, 10_000, 100_000],
    "variations": ["all", "specific"],
}
default_parameters = {
    "model_size": "gpt2",
    "batch_size": 1000,
    "variations": "all",
}


def assert_accuracy(outs1, outs2):
    """Assert accuracy of two outputs."""
    for out1, out2 in zip(outs1[0], outs2[0]):
        assert torch.allclose(out1, out2)
