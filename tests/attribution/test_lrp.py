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

from tdhook.attribution.lrp_helpers.rules import (
    EpsilonRule,
    AlphaBetaRule,
    FlatRule,
    WSquareRule,
    EpsilonPlus,
    PassRule,
    UniformEpsilonRule,
    UniformRule,
    StopRule,
    SoftmaxEpsilonRule,
    raise_for_unconserved_rel_factory,
    RemovableRuleHandle,
    BaseRuleMapper,
)
from tdhook.attribution.lrp_helpers.layers import Sum
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
            rule_mapper=tdhook_mapper,
            init_attr_grads=lambda *_: TensorDict({"output": out_relevance}),
            skip_modules=LRP.default_skip,
            clean_intermediate_keys=False,
        )
        with lrp.prepare(tdhook_module) as hooked_module:
            tdhook_output = hooked_module(TensorDict({"input": tdhook_input}))
            tdhook_in_relevance = tdhook_output.get(("attr", "input"))

        with zennit_composite.context(zennit_module) as modified_module:
            zennit_output = modified_module(zennit_input)
            zennit_output.backward(out_relevance)
            zennit_in_relevance = zennit_input.grad

        torch.testing.assert_close(original_output, tdhook_output.get(("_mod_out", "output")))
        torch.testing.assert_close(tdhook_in_relevance, zennit_in_relevance)

    def test_skip_modules_no_warning(self):
        module = get_sequential_linear_module(seed=0)
        context_factory = LRP(rule_mapper=EpsilonPlus(epsilon=1e-6))
        with pytest.warns(UserWarning, match="No rule found for module"):
            with context_factory.prepare(module):
                pass

        clean_lrp = LRP(rule_mapper=EpsilonPlus(epsilon=1e-6), skip_modules=LRP.default_skip)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            with clean_lrp.prepare(module):
                pass

    def test_removable_rule_handle_module_none(self):
        """Test RemovableRuleHandle.remove() when module reference is None."""

        rule = EpsilonRule(epsilon=1e-6)
        module = get_linear_module(seed=0)

        handle = RemovableRuleHandle(rule, module)
        handle._module_ref = lambda: None

        try:
            handle.remove()
            handle.remove()
        except Exception as exc:
            assert False, f"remove() raised an exception when module is None: {exc}"

        assert handle._rule is rule
        assert handle._module_ref() is None

    def test_pass_rule_forward_errors(self):
        """Test PassRule.forward() error cases."""

        class MultiOutputModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 10)

            def forward(self, *inputs):
                return tuple(self.linear(inp) for inp in inputs)

        rule = PassRule()
        module = MultiOutputModule()
        rule.register(module)

        with pytest.raises(ValueError, match="PassRule requires the number of inputs and outputs to be the same"):

            class FixedOutputModule(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.linear = nn.Linear(10, 10)

                def forward(self, *inputs):
                    return self.linear(inputs[0]), self.linear(inputs[0])

            rule2 = PassRule()
            module2 = FixedOutputModule()
            rule2.register(module2)

            module2(torch.randn(10), torch.randn(10), torch.randn(10))

        with pytest.raises(ValueError, match="Input.*and output.*have different shapes"):

            class ShapeMismatchModule(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.linear1 = nn.Linear(5, 10)
                    self.linear2 = nn.Linear(10, 10)

                def forward(self, x1, x2):
                    return self.linear1(x1), self.linear2(x2)

            rule3 = PassRule()
            module3 = ShapeMismatchModule()
            rule3.register(module3)

            module3(torch.randn(5), torch.randn(10))

        rule.unregister(module)

    def test_uniform_epsilon_rule(self):
        """Test UniformEpsilonRule backward pass."""

        rule = UniformEpsilonRule(epsilon=1e-6)
        module = get_linear_module(seed=0)
        rule.register(module)

        input_tensor = torch.randn(10, requires_grad=True)
        output = module(input_tensor)
        out_relevance = torch.randn_like(output)

        output.backward(out_relevance)

        rule.unregister(module)

    def test_wsquare_rule(self):
        """Test WSquareRule forward and backward passes."""

        rule = WSquareRule(stabilizer=1e-6)
        module = get_linear_module(seed=0)
        rule.register(module)

        input_tensor = torch.randn(10, requires_grad=True)
        output = module(input_tensor)
        out_relevance = torch.randn_like(output)

        # This should work without errors
        output.backward(out_relevance)

        rule.unregister(module)

    def test_alpha_beta_rule_validation(self):
        """Test AlphaBetaRule parameter validation."""

        with pytest.raises(ValueError, match="Both alpha and beta parameters must be non-negative"):
            AlphaBetaRule(alpha=-1.0, beta=1.0)

        with pytest.raises(ValueError, match="Both alpha and beta parameters must be non-negative"):
            AlphaBetaRule(alpha=1.0, beta=-1.0)

        with pytest.raises(ValueError, match="The difference of parameters alpha - beta must equal 1"):
            AlphaBetaRule(alpha=2.0, beta=0.5)

    def test_alpha_beta_rule_multiple_inputs(self):
        """Test AlphaBetaRule with multiple inputs (should raise NotImplementedError)."""

        class MultiInputModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 10)

            def forward(self, *inputs):
                return self.linear(inputs[0])

        rule = AlphaBetaRule(alpha=2.0, beta=1.0)
        module = MultiInputModule()
        rule.register(module)

        with pytest.raises(NotImplementedError, match="AlphaBetaRule does not support multiple inputs"):
            module(torch.randn(10), torch.randn(10))

        rule.unregister(module)

    def test_softmax_epsilon_rule(self):
        """Test SoftmaxEpsilonRule backward pass."""

        rule = SoftmaxEpsilonRule(epsilon=1e-6)
        module = get_linear_module(seed=0)
        rule.register(module)

        input_tensor = torch.randn(10, requires_grad=True)
        output = module(input_tensor)
        out_relevance = torch.randn_like(output)

        # This should work without errors
        output.backward(out_relevance)

        rule.unregister(module)

    def test_uniform_rule(self):
        """Test UniformRule forward and backward passes."""

        rule = UniformRule()
        module = get_linear_module(seed=0)
        rule.register(module)

        input_tensor = torch.randn(10, requires_grad=True)
        output = module(input_tensor)
        out_relevance = torch.randn_like(output)

        # This should work without errors
        output.backward(out_relevance)

        rule.unregister(module)

    def test_stop_rule(self):
        """Test StopRule forward and backward passes."""

        rule = StopRule()
        module = get_linear_module(seed=0)
        rule.register(module)

        input_tensor = torch.randn(10, requires_grad=True)
        output = module(input_tensor)
        out_relevance = torch.randn_like(output)

        # This should work without errors
        output.backward(out_relevance)

        rule.unregister(module)

    def test_base_rule_mapper_call(self):
        """Test BaseRuleMapper._call() method."""

        mapper = BaseRuleMapper()

        activation_module = nn.ReLU()
        rule = mapper._call("relu", activation_module)
        assert rule is not None

        bn_module = nn.BatchNorm1d(10)
        rule = mapper._call("bn", bn_module)
        assert rule is not None

        sum_module = Sum()
        rule = mapper._call("sum", sum_module)
        assert rule is not None

        avgpool_module = nn.AdaptiveAvgPool1d(1)
        rule = mapper._call("avgpool", avgpool_module)
        assert rule is not None

    def test_epsilon_plus_call(self):
        """Test EpsilonPlus._call() method."""

        mapper = EpsilonPlus()

        conv_module = nn.Conv1d(10, 10, 3)
        rule = mapper._call("conv", conv_module)
        assert rule is not None

        linear_module = nn.Linear(10, 10)
        rule = mapper._call("linear", linear_module)
        assert rule is not None

    def test_raise_for_unconserved_rel_factory_error_handling(self):
        """Test raise_for_unconserved_rel_factory error handling."""

        module = get_linear_module(seed=0)

        hook = raise_for_unconserved_rel_factory(atol=1e-10, rtol=1e-10)

        in_relevances = (torch.randn(10), torch.randn(10))
        out_relevances = (torch.randn(10),)

        with pytest.raises(RuntimeError, match="Unconserved relevance"):
            hook(module, in_relevances, out_relevances)

        in_relevances = torch.randn(10)
        out_relevances = torch.randn(10)

        with pytest.raises(RuntimeError, match="Unconserved relevance"):
            hook(module, in_relevances, out_relevances)

    def test_epsilon_rule_no_input_grad(self):
        """Test EpsilonRule when no inputs require gradients."""

        rule = EpsilonRule(epsilon=1e-6)
        module = get_linear_module(seed=0)
        rule.register(module)

        input_tensor = torch.randn(10, requires_grad=False)
        output = module(input_tensor)

        assert not output.requires_grad

        rule.unregister(module)

    def test_epsilon_plus_super_call(self):
        """Test EpsilonPlus._call() when super()._call() is invoked."""

        mapper = EpsilonPlus()

        activation_module = nn.ReLU()
        rule = mapper._call("relu", activation_module)
        assert rule is not None

        class CustomModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.param = nn.Parameter(torch.randn(10))

            def forward(self, x):
                return x + self.param

        custom_module = CustomModule()
        rule = mapper._call("custom", custom_module)
        assert rule is None
