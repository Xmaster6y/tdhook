"""
Input times gradient integration tests
"""

import torch
from tensordict import TensorDict

from tdhook.attribution.input_times_gradient import InputTimesGradient


class TestInputTimesGradient:
    def test_input_times_gradient(self, default_test_model):
        """Test the input times gradient."""
        context = InputTimesGradient()
        with context.prepare(default_test_model) as hooked_module:
            data = TensorDict({"input": torch.randn(2, 3, 10)}, batch_size=[2, 3])
            hooked_module(data)
            assert data["output"].shape == (2, 3, 5)
            assert data["input_attr"].shape == (2, 3, 10)
