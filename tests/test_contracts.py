import pytest

from tdhook.contracts import Access, EffectSpec, GradientMode, KeyContract, MethodSpec, RunKind


def test_key_contract_accepts_nested_tensordict_keys():
    contract = KeyContract(
        in_keys=("input", ("baseline", "input")),
        out_keys=(("attributions", "input"),),
    )

    assert contract.in_keys == ("input", ("baseline", "input"))
    assert contract.out_keys == (("attributions", "input"),)


@pytest.mark.parametrize("key", ("", (), ("valid", 1)))
def test_key_contract_rejects_invalid_tensordict_keys(key):
    with pytest.raises((TypeError, ValueError), match="TensorDict keys"):
        KeyContract(in_keys=(key,))


def test_key_contract_requires_explicit_overwrite():
    with pytest.raises(ValueError, match="overwrite_keys"):
        KeyContract(in_keys=("hidden",), out_keys=("hidden",))

    contract = KeyContract(in_keys=("hidden",), out_keys=("hidden",), overwrite_keys=frozenset({"hidden"}))

    assert contract.overwrite_keys == frozenset({"hidden"})


def test_key_contract_rejects_overlapping_output_paths():
    with pytest.raises(ValueError, match="overlapping TensorDict paths"):
        KeyContract(out_keys=("cache", ("cache", "activation")))


def test_effect_spec_reports_write_conflicts_only():
    reader = EffectSpec(activations=Access.READ, parameters=Access.READ)
    other_reader = EffectSpec(activations=Access.READ)
    writer = EffectSpec(activations=Access.WRITE)

    assert reader.conflict_domains(other_reader) == ()
    assert reader.conflict_domains(writer) == ("activations",)


def test_transform_contract_has_no_model_execution_semantics():
    transform = MethodSpec(
        "summarise",
        keys=KeyContract(in_keys=("activation",), out_keys=("summary",)),
        run_kind=RunKind.TRANSFORM,
        model_passes=0,
        gradient_mode=GradientMode.DISABLED,
    )

    assert transform.model_passes == 0


def test_stateful_transform_can_declare_method_state():
    transform = MethodSpec(
        "fit-estimator",
        run_kind=RunKind.TRANSFORM,
        model_passes=0,
        gradient_mode=GradientMode.DISABLED,
        effects=EffectSpec(state=Access.WRITE),
    )

    assert transform.effects.state is Access.WRITE


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"model_passes": 0}, "at least one model pass"),
        ({"model_passes": True}, "must be an integer"),
        ({"coexecution_key": ""}, "must be non-empty"),
        (
            {"gradient_mode": GradientMode.REQUIRED},
            "must declare a gradient effect",
        ),
    ],
)
def test_method_spec_rejects_incoherent_execution_contracts(kwargs, message):
    with pytest.raises((TypeError, ValueError), match=message):
        MethodSpec("invalid", **kwargs)


def test_compatible_read_only_methods_can_share_one_model_run():
    first = MethodSpec(
        "cache-first",
        keys=KeyContract(in_keys=("input",), out_keys=(("cache", "first"),)),
        effects=EffectSpec(activations=Access.READ),
        coexecution_key="read-hooks-v1",
    )
    second = MethodSpec(
        "cache-second",
        keys=KeyContract(in_keys=("input",), out_keys=(("cache", "second"),)),
        effects=EffectSpec(activations=Access.READ),
        coexecution_key="read-hooks-v1",
    )

    assert first.coexecution_incompatibility(second) is None


def test_coexecution_reports_effect_and_key_conflicts():
    reader = MethodSpec(
        "reader",
        keys=KeyContract(in_keys=("input",), out_keys=(("cache", "activation"),)),
        effects=EffectSpec(activations=Access.READ),
        coexecution_key="hooks-v1",
    )
    writer = MethodSpec(
        "writer",
        keys=KeyContract(in_keys=("input",), out_keys=("output",)),
        effects=EffectSpec(activations=Access.WRITE),
        coexecution_key="hooks-v1",
    )
    colliding_reader = MethodSpec(
        "colliding-reader",
        keys=KeyContract(in_keys=("input",), out_keys=(("cache", "activation"),)),
        effects=EffectSpec(activations=Access.READ),
        coexecution_key="hooks-v1",
    )

    assert reader.coexecution_incompatibility(writer) == "methods have conflicting effects: activations"
    assert "conflicting TensorDict keys" in reader.coexecution_incompatibility(colliding_reader)
