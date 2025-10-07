"""
Integrated gradients
"""

import torch
from typing import List, Optional, Callable, Dict
from tensordict import TensorDict, merge_tensordicts

from tdhook.attribution.gradient_helpers.helpers import approximation_parameters
from tdhook.attribution.gradient_helpers import GradientAttributionWithBaseline
from tdhook._types import UnraveledKey


class IntegratedGradients(GradientAttributionWithBaseline):
    def __init__(
        self,
        use_inputs: bool = True,
        use_outputs: bool = True,
        input_modules: Optional[List[str]] = None,
        target_modules: Optional[List[str]] = None,
        init_attr_targets: Optional[Callable[[TensorDict, TensorDict], TensorDict]] = None,
        init_attr_inputs: Optional[Callable[[TensorDict, TensorDict], TensorDict]] = None,
        init_attr_grads: Optional[Callable[[TensorDict, TensorDict], TensorDict]] = None,
        additional_init_keys: Optional[List[UnraveledKey]] = None,
        output_grad_callbacks: Optional[Dict[str, Callable]] = None,
        attribution_key: UnraveledKey = "attr",
        clean_intermediate_keys: bool = True,
        cache_callback: Optional[Callable] = None,
        compute_convergence_delta: bool = False,
        baseline_key: UnraveledKey = "baseline",
        multiply_by_inputs: bool = False,
        method: str = "gausslegendre",
        n_steps: int = 50,
    ):
        super().__init__(
            use_inputs=use_inputs,
            use_outputs=use_outputs,
            input_modules=input_modules,
            target_modules=target_modules,
            init_attr_targets=init_attr_targets,
            init_attr_inputs=init_attr_inputs,
            init_attr_grads=init_attr_grads,
            additional_init_keys=additional_init_keys,
            output_grad_callbacks=output_grad_callbacks,
            attribution_key=attribution_key,
            clean_intermediate_keys=clean_intermediate_keys,
            cache_callback=cache_callback,
            compute_convergence_delta=compute_convergence_delta,
            baseline_key=baseline_key,
            multiply_by_inputs=multiply_by_inputs,
        )
        self._method = method
        self._n_steps = n_steps

        step_sizes_func, alphas_func = approximation_parameters(self._method)
        step_sizes, alphas = step_sizes_func(self._n_steps), alphas_func(self._n_steps)
        self._step_sizes = step_sizes
        self._alphas = alphas

    def _reduce_baselines_fn(self, td: TensorDict, in_keys: List[UnraveledKey]) -> TensorDict:
        inputs = td.select(*in_keys)
        baselines = td[self._baseline_key]
        additional_init_tensors = td.select(*self._additional_init_keys)

        if self._init_attr_inputs is not None:
            needs_baselines = self._init_attr_inputs(inputs, additional_init_tensors)
            other_inputs = inputs.select(
                *(k for k in inputs.keys(True, True) if k not in needs_baselines.keys(True, True))
            )
        else:
            needs_baselines = inputs
            other_inputs = TensorDict(batch_size=inputs.batch_size)

        new_bs = (*inputs.batch_size, self._n_steps)
        expanded_other_inputs = other_inputs.unsqueeze(-1).expand(new_bs)
        reduced_baselines = torch.stack(
            [baselines + alpha * (needs_baselines - baselines) for alpha in self._alphas], dim=-1
        )
        td["_register_in"] = merge_tensordicts(expanded_other_inputs, reduced_baselines)
        return td

    def _grad_attr(
        self,
        grads: TensorDict,
        inputs: TensorDict,
    ) -> TensorDict:
        steps = torch.tensor(self._step_sizes).float().to(grads.device)
        return grads.mul_(steps).sum(dim=-1)

    @staticmethod
    def init_attr_targets_with_labels(
        outputs: TensorDict,
        additional_init_tensors: TensorDict,
        selected_out_keys: List[UnraveledKey],
        label_key: UnraveledKey = "label",
    ) -> TensorDict:
        targets = outputs.select(*selected_out_keys)
        labels = additional_init_tensors[label_key].unsqueeze(-1).expand(targets.shape)
        d = {}
        for k in targets.keys(True, True):
            one_hot_labels = torch.nn.functional.one_hot(labels[k], num_classes=targets[k].shape[-1]).to(bool)
            if one_hot_labels.shape != targets[k].shape:
                raise ValueError(
                    f"One-hot labels shape {one_hot_labels.shape} does not match target shape {targets[k].shape}"
                )
            d[k] = targets[k][one_hot_labels].reshape(targets.batch_size)
        return TensorDict(d, batch_size=targets.batch_size)
