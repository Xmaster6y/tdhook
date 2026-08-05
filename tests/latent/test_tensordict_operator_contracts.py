import pytest
import torch
from tensordict import TensorDict
from tensordict.nn import TensorDictModuleBase

from tdhook.latent.dimension_estimation import (
    CaPcaDimensionEstimator,
    LocalKnnDimensionEstimator,
    LocalPcaDimensionEstimator,
    TwoNnDimensionEstimator,
)
from tdhook.latent.representation_similarity import CkaEstimator, InformationImbalanceEstimator


@pytest.mark.parametrize(
    "estimator",
    [
        TwoNnDimensionEstimator(in_key=("representations", "samples"), out_key=("metrics", "dimension")),
        LocalKnnDimensionEstimator(
            k=2,
            in_key=("representations", "samples"),
            out_key=("metrics", "dimension"),
        ),
        LocalPcaDimensionEstimator(
            k=2,
            in_key=("representations", "samples"),
            out_key=("metrics", "dimension"),
        ),
        CaPcaDimensionEstimator(
            k=2,
            in_key=("representations", "samples"),
            out_key=("metrics", "dimension"),
        ),
    ],
)
def test_dimension_estimators_are_native_nested_key_tensordict_modules(estimator):
    samples = torch.randn(12, 4)
    data = TensorDict({"representations": {"samples": samples}}, batch_size=[])

    result = estimator(data)

    assert isinstance(estimator, TensorDictModuleBase)
    assert estimator.in_keys == [("representations", "samples")]
    assert estimator.out_keys == [("metrics", "dimension")]
    assert result.get(("metrics", "dimension")) is not None


def test_twonn_auxiliary_outputs_suffix_only_the_nested_leaf():
    estimator = TwoNnDimensionEstimator(
        in_key=("representations", "samples"),
        out_key=("metrics", "dimension"),
        return_xy=True,
    )

    result = estimator(TensorDict({"representations": {"samples": torch.randn(12, 4)}}, batch_size=[]))

    assert estimator.out_keys == [
        ("metrics", "dimension"),
        ("metrics", "dimension_x"),
        ("metrics", "dimension_y"),
    ]
    assert result.get(("metrics", "dimension_x")) is not None
    assert result.get(("metrics", "dimension_y")) is not None


def test_representation_estimators_are_native_nested_key_tensordict_modules():
    x = torch.randn(16, 5)
    y = torch.randn(16, 3)
    data = TensorDict({"representations": {"left": x, "right": y}}, batch_size=[])
    cka = CkaEstimator(
        in_key_a=("representations", "left"),
        in_key_b=("representations", "right"),
        out_key=("metrics", "cka"),
    )
    imbalance = InformationImbalanceEstimator(
        in_key_a=("representations", "left"),
        in_key_b=("representations", "right"),
        out_key_a_to_b=("metrics", "left_to_right"),
        out_key_b_to_a=("metrics", "right_to_left"),
    )

    result = imbalance(cka(data))

    assert cka.in_keys == [("representations", "left"), ("representations", "right")]
    assert cka.out_keys == [("metrics", "cka")]
    assert imbalance.out_keys == [("metrics", "left_to_right"), ("metrics", "right_to_left")]
    assert result.get(("metrics", "cka")).ndim == 0
    assert result.get(("metrics", "left_to_right")).ndim == 0
    assert result.get(("metrics", "right_to_left")).ndim == 0
