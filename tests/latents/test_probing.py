"""
Tests for the latents functionality.
"""

import torch.nn as nn

from tdhook.acteng.linear_probing import ProbingContext


class TestProbingContext:
    """Test the ProbingContext class."""

    def test_probing_context_creation(self):
        """Test creating a ProbingContext."""

        def get_probe(key):
            return nn.Linear(20, 1)

        context = ProbingContext("linear\\d+", get_probe)
        assert isinstance(context, ProbingContext)
        assert context._get_probe == get_probe

    def test_probing_context_get_probe(self):
        """Test the get_probe function."""

        def get_probe(key):
            return nn.Linear(20, 1)

        context = ProbingContext("linear\\d+", get_probe)
        probe = context._get_probe("linear1")
        assert isinstance(probe, nn.Linear)
        assert probe.in_features == 20
        assert probe.out_features == 1
