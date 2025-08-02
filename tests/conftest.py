"""
Global fixtures for the tests.
"""

import pytest
import torch.nn as nn


class TestModel(nn.Module):
    """Simple test model for tdhook testing."""

    def __init__(self, input_size=10, hidden_size=20, output_size=5):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.activation(self.linear1(x))
        x = self.activation(self.linear2(x))
        x = self.linear3(x)
        return x


@pytest.fixture(scope="function")
def default_test_model():
    """Fixture providing a simple test model."""
    return TestModel()


@pytest.fixture
def get_model():
    """Fixture providing a function to create models with custom parameters."""

    def _get_model(*, input_size=10, hidden_size=20, output_size=5):
        return TestModel(input_size, hidden_size, output_size)

    return _get_model
