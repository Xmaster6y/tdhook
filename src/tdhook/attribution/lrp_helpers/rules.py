"""
LRP rules.

This code is adapted from the Zennit library (LGPL-3.0) and the LXT library (Clear BSD)
Original sources:
- https://github.com/chr5tphr/zennit/blob/main/src/zennit/rules.py
- https://github.com/rachtibat/LRP-eXplains-Transformers/blob/main/lxt/explicit/rules.py
"""

from abc import ABCMeta, abstractmethod
from contextlib import contextmanager
from typing import Callable, List, Optional
import weakref

import torch
import torch.nn as nn
from torch.autograd.function import Function, FunctionMeta

from .types import Activation, AvgPool, BatchNorm, Convolution, Linear, MaxPool
from .layers import Sum


def stabilize(tensor, epsilon=1e-6):
    return tensor + epsilon * ((-1) ** (tensor < 0))


class ParamModifier:
    def __init__(
        self,
        modify_fn: Optional[Callable[[str, nn.Parameter], torch.Tensor]] = None,
        select_fn: Optional[Callable[[str, nn.Parameter], bool]] = None,
    ):
        self._modify_fn = modify_fn or (lambda x, _: x)
        self._select_fn = select_fn or (lambda _, __: False)

    def state_dicts(self, module: nn.Module):
        original_state = {
            name: param for name, param in module.named_parameters(recurse=False) if self._select_fn(name, param)
        }
        modified_state = {
            name: self._modify_fn(name, param)
            for name, param in module.named_parameters(recurse=False)
            if self._select_fn(name, param)
        }
        return original_state, modified_state

    @contextmanager
    def __call__(self, module: nn.Module):
        original_state = {}
        try:
            original_state, modified_state = self.state_dicts(module)
            module.load_state_dict(modified_state, strict=False, assign=True)
            yield module
        finally:
            module.load_state_dict(original_state, strict=False, assign=True)

    @staticmethod
    def from_modifiers(modifiers: List["ParamModifier"]):
        def select_fn(name, param):
            return any(modifier._select_fn(name, param) for modifier in modifiers)

        def modify_fn(name, param):
            new_param = param
            for modifier in modifiers:
                if modifier._select_fn(name, new_param):
                    new_param = modifier._modify_fn(name, new_param)
            return new_param

        return ParamModifier(modify_fn=modify_fn, select_fn=select_fn)

    @staticmethod
    def select_all(name: str, param: nn.Parameter):
        return True


class RemovableRuleHandle:
    def __init__(self, rule: "Rule", module: nn.Module):
        self._rule = rule
        self._module_ref = weakref.ref(module)

    def remove(self):
        module = self._module_ref()
        if module is not None:
            self._rule.unregister(module)


class AbstractFunctionMeta(ABCMeta, FunctionMeta):
    pass


class Rule(Function, metaclass=AbstractFunctionMeta):
    def __init__(self):
        self._apply_kwargs = {}
        # TODO: Add zero_params argument for all rules

    def register(self, module: nn.Module):
        module._prev_forward = module.forward

        def forward(*inputs, **model_kwargs):
            nonlocal module
            return self.apply(self._apply_kwargs, module, model_kwargs, *inputs)

        module.forward = forward
        return RemovableRuleHandle(self, module)

    def unregister(self, module: nn.Module):
        module.forward = module._prev_forward
        del module._prev_forward

    @staticmethod
    @abstractmethod
    def forward(ctx, apply_kwargs, module, model_kwargs, *inputs):
        pass

    @staticmethod
    @abstractmethod
    def backward(ctx, *out_relevance):
        pass


class EpsilonRule(Rule):
    def __init__(self, epsilon=1e-6):
        super().__init__()
        self._apply_kwargs["epsilon"] = epsilon

    @property
    def epsilon(self):
        return self._apply_kwargs["epsilon"]

    @epsilon.setter
    def epsilon(self, value):
        self._apply_kwargs["epsilon"] = value

    @staticmethod
    def forward(ctx, apply_kwargs, module, model_kwargs, *inputs):
        # TODO: Move logic to backward
        if not any(ctx.needs_input_grad):
            return module._prev_forward(*inputs, **model_kwargs)

        # TODO: Check if really needed
        inputs = tuple(inp.detach().requires_grad_() if inp.requires_grad else inp for inp in inputs)
        with torch.enable_grad():
            output = module._prev_forward(*inputs, **model_kwargs)

        ctx.save_for_backward(output, *inputs)
        ctx.epsilon = apply_kwargs["epsilon"]
        return output.detach()

    @staticmethod
    def backward(ctx, *out_relevance):
        output, *inputs = ctx.saved_tensors
        relevance_norm = out_relevance[0] / stabilize(output, ctx.epsilon)
        grads = torch.autograd.grad(output, inputs, relevance_norm)
        return (
            None,
            None,
            None,
            *(grads[i].mul(inputs[i]) if ctx.needs_input_grad[i + 3] else None for i in range(len(inputs))),
        )


class UniformEpsilonRule(EpsilonRule):
    @staticmethod
    def backward(ctx, *out_relevance):
        output, *inputs = ctx.saved_tensors
        relevance_norm = out_relevance[0] / stabilize(output, ctx.epsilon) / len(inputs)
        grads = torch.autograd.grad(output, inputs, relevance_norm)
        return (
            None,
            None,
            None,
            *(grads[i].mul(inputs[i]) if ctx.needs_input_grad[i + 3] else None for i in range(len(inputs))),
        )


class PassRule(Rule):
    @staticmethod
    def forward(ctx, apply_kwargs, module, model_kwargs, *inputs):
        # TODO: Move logic to backward
        n_inputs = len(inputs)
        output = module._prev_forward(*inputs, **model_kwargs)
        n_outputs = len(output) if isinstance(output, tuple) else 1
        if n_inputs != n_outputs:
            raise ValueError(
                (
                    "PassRule requires the number of inputs and outputs to be the same, ",
                    f"got {n_inputs} inputs and {n_outputs} outputs",
                )
            )
        all_outputs = [output] if isinstance(output, torch.Tensor) else output
        for index, (i, o) in enumerate(zip(inputs, all_outputs)):
            if i.shape != o.shape:
                raise ValueError(
                    f"Input (shape={i.shape}) and output (shape={o.shape}) have different shapes at index {index}"
                )
        return output

    @staticmethod
    def backward(ctx, *out_relevance):
        return None, None, None, *out_relevance


class IgnoreRule(Rule):
    def register(self, module: nn.Module):
        return RemovableRuleHandle(self, module)

    def unregister(self, module: nn.Module):
        pass

    @staticmethod
    def forward(ctx, apply_kwargs, module, model_kwargs, *inputs):
        pass

    @staticmethod
    def backward(ctx, *out_relevance):
        pass


class WSquareRule(Rule):
    def __init__(self, stabilizer=1e-6):
        super().__init__()
        self._apply_kwargs["_modifier"] = ParamModifier(
            select_fn=ParamModifier.select_all, modify_fn=lambda _, param: param**2
        )
        self._apply_kwargs["stabilizer"] = stabilizer

    @staticmethod
    def forward(ctx, apply_kwargs, module, model_kwargs, *inputs):
        # TODO: Move logic to backward
        ctx.stabilizer = apply_kwargs["stabilizer"]
        output = module._prev_forward(*inputs, **model_kwargs)
        mod_inputs = tuple(torch.ones_like(inp).requires_grad_() for inp in inputs)
        with torch.enable_grad():
            with apply_kwargs["_modifier"](module) as modified_module:
                mod_output = modified_module._prev_forward(*mod_inputs, **model_kwargs)
        ctx.save_for_backward(mod_output, *mod_inputs)
        return output

    @staticmethod
    def backward(ctx, *out_relevance):
        mod_output, *mod_inputs = ctx.saved_tensors
        normed_relevance = out_relevance[0] / stabilize(mod_output, ctx.stabilizer)
        in_relevance = torch.autograd.grad(mod_output, mod_inputs, normed_relevance)
        return None, None, None, *in_relevance


class FlatRule(WSquareRule):
    def __init__(self, stabilizer=1e-6):
        super().__init__(stabilizer)
        to_ones = ParamModifier(select_fn=ParamModifier.select_all, modify_fn=lambda _, param: torch.ones_like(param))
        zero_bias = ParamModifier(
            select_fn=lambda name, param: name == "bias", modify_fn=lambda _, param: torch.zeros_like(param)
        )
        self._apply_kwargs["_modifier"] = ParamModifier.from_modifiers([to_ones, zero_bias])


class UniformRule(Rule):
    @staticmethod
    def forward(ctx, apply_kwargs, module, model_kwargs, *inputs):
        ctx.n_inputs = len(inputs)
        return module._prev_forward(*inputs, **model_kwargs)

    @staticmethod
    def backward(ctx, *out_relevances):
        return None, None, None, *[out_relevances[0] / ctx.n_inputs for _ in range(ctx.n_inputs)]


class StopRule(Rule):
    @staticmethod
    def forward(ctx, apply_kwargs, module, model_kwargs, *inputs):
        ctx.n_inputs = len(inputs)
        return module._prev_forward(*inputs, **model_kwargs)

    @staticmethod
    def backward(ctx, *out_relevances):
        return None, None, None, *(None,) * ctx.n_inputs


class AlphaBetaRule(Rule):
    def __init__(self, alpha=2.0, beta=1.0, stabilizer=1e-6):
        if alpha < 0 or beta < 0:
            raise ValueError("Both alpha and beta parameters must be non-negative!")
        if (alpha - beta) != 1.0:
            raise ValueError("The difference of parameters alpha - beta must equal 1!")

        super().__init__()
        self._apply_kwargs["alpha"] = alpha
        self._apply_kwargs["beta"] = beta
        self._apply_kwargs["stabilizer"] = stabilizer

        self._apply_kwargs["_zero_bias"] = ParamModifier(
            select_fn=lambda name, param: name == "bias", modify_fn=lambda _, param: torch.zeros_like(param)
        )
        self._apply_kwargs["_positive"] = ParamModifier(
            select_fn=ParamModifier.select_all, modify_fn=lambda _, param: param.clamp(min=0)
        )
        self._apply_kwargs["_negative"] = ParamModifier(
            select_fn=ParamModifier.select_all, modify_fn=lambda _, param: param.clamp(max=0)
        )

    @staticmethod
    def forward(ctx, apply_kwargs, module, model_kwargs, *inputs):
        # TODO: Move logic to backward
        output = module._prev_forward(*inputs, **model_kwargs)

        if len(inputs) > 1:
            raise NotImplementedError("AlphaBetaRule does not support multiple inputs")

        pos_input = inputs[0].clamp(min=0).detach().requires_grad_()
        neg_input = inputs[0].clamp(max=0).detach().requires_grad_()
        with torch.enable_grad():
            with apply_kwargs["_positive"](module) as positive_module:
                out_pos = positive_module._prev_forward(pos_input, **model_kwargs)
                with apply_kwargs["_zero_bias"](positive_module) as modified_module:
                    out_pos_zero = modified_module._prev_forward(neg_input, **model_kwargs)
            with apply_kwargs["_negative"](module) as negative_module:
                out_neg = negative_module._prev_forward(pos_input, **model_kwargs)
                with apply_kwargs["_zero_bias"](negative_module) as modified_module:
                    out_neg_zero = modified_module._prev_forward(neg_input, **model_kwargs)

        ctx.save_for_backward(out_pos, out_pos_zero, out_neg, out_neg_zero, neg_input, pos_input)
        ctx.alpha = apply_kwargs["alpha"]
        ctx.beta = apply_kwargs["beta"]
        ctx.stabilizer = apply_kwargs["stabilizer"]
        return output

    @staticmethod
    def backward(ctx, *out_relevance):
        out_pos, out_pos_zero, out_neg, out_neg_zero, neg_input, pos_input = ctx.saved_tensors
        relevance_pos = out_relevance[0] / stabilize(out_pos + out_neg_zero, ctx.stabilizer)
        relevance_neg = out_relevance[0] / stabilize(out_neg + out_pos_zero, ctx.stabilizer)

        pos_grad = torch.autograd.grad(out_pos, pos_input, relevance_pos)
        pos_zero_grad = torch.autograd.grad(out_pos_zero, neg_input, relevance_neg)
        neg_grad = torch.autograd.grad(out_neg, pos_input, relevance_neg)
        neg_zero_grad = torch.autograd.grad(out_neg_zero, neg_input, relevance_pos)

        in_relevance = ctx.alpha * (pos_input * pos_grad[0] + neg_input * neg_zero_grad[0]) - ctx.beta * (
            pos_input * neg_grad[0] + neg_input * pos_zero_grad[0]
        )

        return None, None, None, in_relevance


class SoftmaxEpsilonRule(EpsilonRule):
    @staticmethod
    def backward(ctx, *out_relevances):
        output, *inputs = ctx.saved_tensors

        relevance = (out_relevances[0] - (output * out_relevances[0].sum(-1, keepdim=True))) * inputs[0]

        return None, None, None, relevance


class BaseRuleMapper:
    def __init__(self, stabilizer=1e-6, rule_mapper: Optional[Callable[[str, nn.Module], Rule | None]] = None):
        self._stabilizer = stabilizer
        self._rule_mapper = rule_mapper or (lambda name, module: None)

        self._rules = {
            "pass": PassRule(),
            "norm": EpsilonRule(epsilon=self._stabilizer),
            "ignore": IgnoreRule(),
        }

    def _call(self, name: str, module: nn.Module) -> Rule | None:
        if isinstance(module, (Activation, BatchNorm)):
            return self._rules["pass"]
        if isinstance(module, (MaxPool, nn.Identity, nn.Dropout, nn.Flatten)):
            return self._rules["ignore"]
        elif isinstance(module, (Sum, AvgPool)):
            return self._rules["norm"]

    def __call__(self, name: str, module: nn.Module) -> Rule | None:
        rule = self._rule_mapper(name, module)
        if rule is None:
            return self._call(name, module)
        return rule


class EpsilonPlus(BaseRuleMapper):
    def __init__(
        self, epsilon=1e-6, stabilizer=1e-6, rule_mapper: Optional[Callable[[str, nn.Module], Rule | None]] = None
    ):
        super().__init__(stabilizer, rule_mapper)
        self._rules["epsilon"] = EpsilonRule(epsilon=epsilon)
        self._rules["zplus"] = AlphaBetaRule(alpha=1.0, beta=0.0, stabilizer=stabilizer)

    def _call(self, name: str, module: nn.Module) -> Rule | None:
        if isinstance(module, Convolution):
            return self._rules["zplus"]
        elif isinstance(module, Linear):
            return self._rules["epsilon"]
        return super()._call(name, module)


def raise_for_unconserved_rel_factory(atol: float = 1e-6, rtol: float = 1e-6):
    def raise_for_unconserved_rel(module, in_relevances, out_relevances):
        if isinstance(in_relevances, tuple):
            in_rel_sum = sum(in_rel.sum() for in_rel in in_relevances)
        else:
            in_rel_sum = in_relevances.sum()
        if isinstance(out_relevances, tuple):
            out_rel_sum = sum(out_rel.sum() for out_rel in out_relevances)
        else:
            out_rel_sum = out_relevances.sum()
        if not torch.isclose(in_rel_sum, out_rel_sum, atol=atol, rtol=rtol):
            raise RuntimeError(
                (f"Unconserved relevance for module {module.__class__.__name__} ({in_rel_sum=}) ({out_rel_sum=})")
            )

    return raise_for_unconserved_rel


# TODO: Add lxt rules and tests
