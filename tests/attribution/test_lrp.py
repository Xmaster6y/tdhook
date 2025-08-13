"""
Test LRP rules.
"""

import torch
import torch.nn as nn
import pytest
from tensordict import TensorDict
import warnings

from zennit.rules import Epsilon, AlphaBeta, ZPlus, Flat, WSquare, Pass, Norm
from zennit.composites import EpsilonPlus as ZennitEpsilonPlus

from tdhook.attribution.lrp.rules import (
    EpsilonRule,
    AlphaBetaRule,
    FlatRule,
    WSquareRule,
    EpsilonPlus,
    PassRule,
    raise_for_unconserved_rel_factory,
)
from tdhook.attribution import LRP


def get_linear_module(seed: int):
    torch.manual_seed(seed)
    return torch.nn.Linear(10, 10)


def get_conv_module(seed: int):
    torch.manual_seed(seed)
    return nn.Conv1d(10, 10, 3, padding=1)


def get_sequential_linear_module(seed: int):
    torch.manual_seed(seed)
    return nn.Sequential(
        nn.Linear(10, 10),
        nn.ReLU(),
        nn.Linear(10, 10),
        nn.ReLU(),
    )


def get_sequential_conv_module(seed: int):
    torch.manual_seed(seed)
    return nn.Sequential(
        nn.Conv1d(10, 10, 3, padding=1),
        nn.ReLU(),
        nn.Conv1d(10, 10, 3, padding=1),
        nn.ReLU(),
    )


class TestRules:
    def test_register_unregister(self):
        linear_module = get_linear_module(seed=0)
        rule = EpsilonRule(epsilon=1e-6)
        prev_forward = linear_module.forward
        rule.register(linear_module)
        assert linear_module.forward != prev_forward
        rule.unregister(linear_module)
        assert linear_module.forward == prev_forward

    @pytest.mark.parametrize(
        "factory",
        (
            (get_linear_module, (10,)),
            (get_conv_module, (10, 5)),
        ),
    )
    def test_raise_for_unconserved_rel(self, factory):
        get_module, input_shape = factory
        tdhook_module = get_module(seed=0)
        tdhook_input = torch.randn(*input_shape, requires_grad=True)
        out_relevance = torch.randn_like(tdhook_module(tdhook_input))

        for name, param in tdhook_module.named_parameters(recurse=True):
            if name.endswith("bias"):
                param.data.zero_()

        tdhook_module.register_full_backward_hook(raise_for_unconserved_rel_factory())

        with pytest.raises(RuntimeError):
            tdhook_module(tdhook_input).backward(out_relevance)

        PassRule().register(tdhook_module)
        tdhook_module(tdhook_input).backward(out_relevance)

    @pytest.mark.parametrize(
        "rule",
        (
            EpsilonRule(epsilon=0.0),
            AlphaBetaRule(alpha=2.0, beta=1.0, stabilizer=0.0),
            FlatRule(stabilizer=0.0),
            WSquareRule(stabilizer=0.0),
            PassRule(),
        ),
    )
    @pytest.mark.parametrize(
        "use_bw",
        (True, False),
    )
    @pytest.mark.parametrize(
        "factory",
        (
            (get_linear_module, (10,)),
            (get_conv_module, (10, 5)),
        ),
    )
    def test_return_type_and_conservation(self, factory, rule, use_bw):
        get_module, input_shape = factory
        tdhook_module: nn.Module = get_module(seed=0)
        tdhook_input = torch.randn(*input_shape, requires_grad=True)
        out_relevance = torch.randn_like(tdhook_module(tdhook_input))

        for name, param in tdhook_module.named_parameters(recurse=True):
            if name.endswith("bias"):
                param.data.zero_()
        rule.register(tdhook_module)

        if use_bw:
            tdhook_module(tdhook_input).backward(out_relevance)
            in_relevance = tdhook_input.grad
            assert isinstance(in_relevance, torch.Tensor)
            in_rel_sum = in_relevance.sum()
        else:
            in_relevance = torch.autograd.grad(tdhook_module(tdhook_input), tdhook_input, out_relevance)
            assert isinstance(in_relevance, tuple)
            in_rel_sum = in_relevance[0].sum()

        # TODO: check why tolerance is needed in CI (epsilon+conv)
        torch.testing.assert_close(in_rel_sum, out_relevance.sum(), atol=1e-3, rtol=1e-3)

    @pytest.mark.parametrize(
        "rules",
        (
            (Epsilon(epsilon=1e-6), EpsilonRule(epsilon=1e-6)),
            (AlphaBeta(alpha=2.0, beta=1.0, stabilizer=1e-6), AlphaBetaRule(alpha=2.0, beta=1.0, stabilizer=1e-6)),
            (AlphaBeta(alpha=1.0, beta=0.0, stabilizer=1e-6), AlphaBetaRule(alpha=1.0, beta=0.0, stabilizer=1e-6)),
            (AlphaBeta(alpha=1.0, beta=0.0, stabilizer=1e-6), ZPlus(stabilizer=1e-6)),
            (Flat(stabilizer=1e-6), FlatRule(stabilizer=1e-6)),
            (WSquare(stabilizer=1e-6), WSquareRule(stabilizer=1e-6)),
            (Pass(), PassRule()),
            (Norm(stabilizer=1e-6), EpsilonRule(epsilon=1e-6)),
        ),
    )
    @pytest.mark.parametrize(
        "use_bw",
        (True, False),
    )
    @pytest.mark.parametrize(
        "factory",
        (
            (get_linear_module, (10,)),
            (get_conv_module, (10, 5)),
        ),
    )
    def test_zennit_rules_equivalence(self, factory, rules, use_bw):
        get_module, input_shape = factory
        tdhook_module = get_module(seed=0)
        tdhook_input = torch.randn(*input_shape, requires_grad=True)
        zennit_module = get_module(seed=0)
        zennit_input = torch.randn(*input_shape, requires_grad=True)
        out_relevance = torch.randn_like(tdhook_module(tdhook_input))

        zennit_rule = rules[0]
        zennit_rule.register(zennit_module)
        if use_bw:
            zennit_module(zennit_input).backward(out_relevance)
            zennit_in_relevance = zennit_input.grad
        else:
            zennit_in_relevance = torch.autograd.grad(zennit_module(zennit_input), zennit_input, out_relevance)

        tdhook_rule = rules[1]
        tdhook_rule.register(tdhook_module)
        if use_bw:
            tdhook_module(tdhook_input).backward(out_relevance)
            tdhook_in_relevance = tdhook_input.grad
        else:
            tdhook_in_relevance = torch.autograd.grad(tdhook_module(tdhook_input), tdhook_input, out_relevance)

        torch.testing.assert_close(tdhook_in_relevance, zennit_in_relevance)

    @pytest.mark.parametrize(
        "matching",
        ((ZennitEpsilonPlus(epsilon=1e-6), EpsilonPlus(epsilon=1e-6)),),
    )
    @pytest.mark.parametrize(
        "factory",
        (
            (get_sequential_linear_module, (10,)),
            (get_sequential_conv_module, (10, 5)),
        ),
    )
    def test_lrp_with_sequential_module(self, matching, factory):
        get_module, input_shape = factory
        zennit_composite, tdhook_mapper = matching
        tdhook_module = get_module(seed=4)
        tdhook_input = torch.randn(*input_shape, requires_grad=True)
        zennit_module = get_module(seed=4)
        zennit_input = torch.randn(*input_shape, requires_grad=True)
        out_relevance = torch.randn_like(tdhook_module(tdhook_input))

        original_output = tdhook_module(tdhook_input)

        lrp = LRP(
            rule_mapper=tdhook_mapper, init_grad=lambda _: out_relevance, skip_modules=LRP.skip_root_and_modulelist
        )
        with lrp.prepare(tdhook_module) as hooked_module:
            tdhook_output = hooked_module(TensorDict({"input": tdhook_input}))
            tdhook_in_relevance = tdhook_output.get("input_attr")

        with zennit_composite.context(zennit_module) as modified_module:
            zennit_output = modified_module(zennit_input)
            zennit_output.backward(out_relevance)
            zennit_in_relevance = zennit_input.grad

        torch.testing.assert_close(original_output, tdhook_output.get("output"))
        torch.testing.assert_close(tdhook_in_relevance, zennit_in_relevance)

    def test_skip_modules_no_warning(self):
        module = get_sequential_linear_module(seed=0)
        context_factory = LRP(rule_mapper=EpsilonPlus(epsilon=1e-6))
        with pytest.warns(UserWarning, match="No rule found for module"):
            with context_factory.prepare(module):
                pass

        clean_lrp = LRP(rule_mapper=EpsilonPlus(epsilon=1e-6), skip_modules=LRP.skip_root_and_modulelist)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            with clean_lrp.prepare(module):
                pass
