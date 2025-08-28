import torch

from tdhook.artifacts import Circuit


class TestCircuit:
    def test_circuit_creation(self):
        """Test creating a Circuit with boolean tensors."""
        data = {
            "weight": torch.randn(10, 10) > 0,
            "bias": torch.randn(10) > 0,
        }
        circuit = Circuit(data)
        assert circuit.dtype == torch.bool

    def test_circuit_merge(self):
        """Test merging circuits."""
        circuit_1 = Circuit(
            {
                "weight": torch.randn(10, 10) > 0,
                "bias": torch.randn(10) > 0,
            }
        )

        circuit_2 = Circuit(
            {
                "weight": torch.randn(10, 10) > 0,
                "bias": torch.randn(10) > 0,
            }
        )

        circuit = Circuit.merge([circuit_1, circuit_2])
        assert torch.all(circuit["weight"] == (circuit_1["weight"] | circuit_2["weight"]))
        assert torch.all(circuit["bias"] == (circuit_1["bias"] | circuit_2["bias"]))
