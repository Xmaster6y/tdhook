"""Small helpers used by the ROME reproduction notebook."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from torch import Tensor, nn

from tdhook.latent import SteeringVectors
from tdhook.session import HookSession
from tdhook.targets import Target
from tdhook.workflow import Workflow


@dataclass(frozen=True)
class CausalTraceConfig:
    """The released ROME tracing defaults used by this reproduction."""

    samples: int = 10
    noise_seed: int = 1
    noise_level: float = 0.1
    replace_noise: bool = False


DEFAULT_TRACE_CONFIG = CausalTraceConfig()


def _logits(output: object) -> Tensor:
    if isinstance(output, Mapping):
        value = output["logits"]
    else:
        value = output.logits  # type: ignore[attr-defined]
    if not isinstance(value, Tensor):
        raise TypeError("model output logits must be a tensor")
    return value


def _corrupt_subject(value: Tensor, *, seed: int, noise_level: float, replace: bool) -> Tensor:
    """Match ROME's NumPy RandomState Gaussian corruption, preserving batch item 0."""

    result = value.clone()
    random = np.random.RandomState(seed)
    noise = torch.from_numpy(random.randn(result.shape[0] - 1, *result.shape[1:])).to(result.device)
    noise = noise_level * noise
    if replace:
        result[1:] = noise
    else:
        result[1:] += noise
    return result


def _restore_clean(value: Tensor) -> Tensor:
    result = value.clone()
    result[1:] = result[0]
    return result


def causal_trace_workflow(
    states_to_patch: Sequence[tuple[int, str]],
    answer_token: int,
    subject_range: tuple[int, int],
    *,
    embedding_path: str = "transformer.wte",
    state_output_paths: Mapping[str, tuple[int | str, ...]] | None = None,
    config: CausalTraceConfig = DEFAULT_TRACE_CONFIG,
) -> Workflow:
    """Declare one corrupt-and-restore causal-tracing experiment.

    The first batch item remains the clean reference. The configured TDHook
    method corrupts subject embeddings in the remaining items and restores
    selected hidden states from the clean item. A TensorDict operator then
    reduces the model output to the answer probability consumed by the sweep.
    """

    start, stop = subject_range
    patches: dict[str, list[int]] = defaultdict(list)
    for token, module_path in states_to_patch:
        patches[module_path].append(token)

    embedding = Target(embedding_path, "activation", 1, tuple(range(start, stop)))
    states = tuple(
        Target(
            module_path,
            "activation",
            1,
            tuple(sorted(set(tokens))),
            output_path=(state_output_paths or {}).get(module_path, ()),
        )
        for module_path, tokens in patches.items()
    )

    def intervene(*, module_key: str, output: Tensor) -> Tensor:
        if module_key == embedding_path:
            return _corrupt_subject(
                output,
                seed=config.noise_seed,
                noise_level=config.noise_level,
                replace=config.replace_noise,
            )
        return _restore_clean(output)

    def answer_probability(output: object) -> Tensor:
        probabilities = torch.softmax(_logits(output)[1:, -1, :], dim=-1)
        return probabilities[:, answer_token].mean()

    return Workflow(
        SteeringVectors([embedding, *states], intervene),
        TensorDictModule(answer_probability, in_keys=["output"], out_keys=[("metrics", "answer_probability")]),
    )


def trace_with_patch_tdhook(
    model: nn.Module,
    model_inputs: Mapping[str, Tensor],
    states_to_patch: Sequence[tuple[int, str]],
    answer_token: int,
    subject_range: tuple[int, int],
    *,
    embedding_path: str = "transformer.wte",
    state_output_paths: Mapping[str, tuple[int | str, ...]] | None = None,
    config: CausalTraceConfig = DEFAULT_TRACE_CONFIG,
) -> Tensor:
    """Run one ROME-compatible causal trace using public TDHook operations.

    Batch item zero is clean.  The remaining items receive identical seeded
    corruption semantics to ``experiments/causal_trace.py`` in the pinned ROME
    revision, then selected token states are restored from item zero.
    """

    if next(iter(model_inputs.values())).shape[0] != config.samples + 1:
        raise ValueError("model input batch must contain one clean item followed by config.samples copies")
    start, stop = subject_range
    if not 0 <= start < stop:
        raise ValueError("subject_range must be a non-empty half-open range")

    model_module = TensorDictModule(
        model,
        in_keys={key: key for key in model_inputs},
        out_keys=["output"],
    )
    artifacts = TensorDict(dict(model_inputs), batch_size=[])
    workflow = causal_trace_workflow(
        states_to_patch,
        answer_token,
        subject_range,
        embedding_path=embedding_path,
        state_output_paths=state_output_paths,
        config=config,
    )
    with torch.no_grad():
        result = workflow(model_module, artifacts)
    return result[("metrics", "answer_probability")]


def causal_trace_grid(
    model: nn.Module,
    model_inputs: Mapping[str, Tensor],
    answer_token: int,
    subject_range: tuple[int, int],
    layer_paths: Sequence[str],
    *,
    embedding_path: str = "transformer.wte",
    token_indices: Iterable[int] | None = None,
    config: CausalTraceConfig = DEFAULT_TRACE_CONFIG,
) -> Tensor:
    """Restore each token/layer state and return its answer probability."""

    token_indices = tuple(
        token_indices if token_indices is not None else range(next(iter(model_inputs.values())).shape[1])
    )
    rows = []
    for token in token_indices:
        rows.append(
            torch.stack(
                [
                    trace_with_patch_tdhook(
                        model,
                        model_inputs,
                        [(token, layer_path)],
                        answer_token,
                        subject_range,
                        embedding_path=embedding_path,
                        state_output_paths={layer_path: (0,)},
                        config=config,
                    )
                    for layer_path in layer_paths
                ]
            )
        )
    return torch.stack(rows)


def causal_trace_window_grid(
    model: nn.Module,
    model_inputs: Mapping[str, Tensor],
    answer_token: int,
    subject_range: tuple[int, int],
    layer_paths: Sequence[str],
    *,
    component: str,
    window: int = 10,
    embedding_path: str = "transformer.wte",
    token_indices: Iterable[int] | None = None,
    config: CausalTraceConfig = DEFAULT_TRACE_CONFIG,
) -> Tensor:
    """Match ROME's sliding-window MLP or attention restoration sweep."""

    if component not in {"mlp", "attn"}:
        raise ValueError("component must be 'mlp' or 'attn'")
    if isinstance(window, bool) or not isinstance(window, int) or window <= 0:
        raise ValueError("window must be a positive integer")
    token_indices = tuple(
        token_indices if token_indices is not None else range(next(iter(model_inputs.values())).shape[1])
    )
    rows = []
    for token in token_indices:
        row = []
        for center in range(len(layer_paths)):
            start = max(0, center - window // 2)
            stop = min(len(layer_paths), center - (-window // 2))
            states = [(token, f"{layer_paths[layer]}.{component}") for layer in range(start, stop)]
            row.append(
                trace_with_patch_tdhook(
                    model,
                    model_inputs,
                    states,
                    answer_token,
                    subject_range,
                    embedding_path=embedding_path,
                    state_output_paths={path: (() if component == "mlp" else (0,)) for _, path in states},
                    config=config,
                )
            )
        rows.append(torch.stack(row))
    return torch.stack(rows)


def rank_one_update(left: Tensor, right: Tensor, weight: Tensor) -> Tensor:
    """Return ROME's outer product in the storage orientation of ``weight``."""

    update = left.unsqueeze(1) @ right.unsqueeze(0)
    if update.shape == weight.shape:
        return update
    if update.T.shape == weight.shape:
        return update.T
    raise ValueError("rank-one update does not match the edited weight in either orientation")


@contextmanager
def temporary_rank_one_edit(
    model: nn.Module,
    module_path: str,
    parameter: str,
    left: Tensor,
    right: Tensor,
):
    """Expose an edited model only inside the context and restore it on every exit."""

    module = Target(module_path, "parameter", 0, (0,), parameter=parameter).validate(model)
    weight = module.get_parameter(parameter)
    edited_weight = weight.detach() + rank_one_update(left, right, weight)
    whole_weight = Target(module_path, "parameter", 0, tuple(range(weight.shape[0])), parameter=parameter)
    with HookSession(model) as session:
        session.replace(whole_weight, edited_weight)
        yield model


def case_score(metrics: Mapping[str, Any]) -> dict[str, float]:
    """Reduce official CounterFact probability metrics using its NLL ordering."""

    def success(items: Sequence[Mapping[str, float]], *, prefer_new: bool) -> float:
        comparisons = [
            item["target_new"] < item["target_true"] if prefer_new else item["target_true"] < item["target_new"]
            for item in items
        ]
        return float(np.mean(comparisons))

    return {
        "rewrite_efficacy": success(metrics["rewrite_prompts_probs"], prefer_new=True),
        "paraphrase_generalization": success(metrics["paraphrase_prompts_probs"], prefer_new=True),
        "neighborhood_specificity": success(metrics["neighborhood_prompts_probs"], prefer_new=False),
    }


def summarize_counterfact(cases: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Average the three CounterFact measures over a list of edited cases."""

    scores = [case_score(case["post"]) for case in cases]
    return {
        "number_cases": len(cases),
        "metrics": {
            metric: float(np.mean([score[metric] for score in scores]))
            for metric in ("rewrite_efficacy", "paraphrase_generalization", "neighborhood_specificity")
        },
    }


def parity_report(tdhook: Tensor, official: Tensor, *, atol: float, rtol: float) -> dict[str, Any]:
    """Record, rather than merely assert, fixed-case numerical parity."""

    difference = (tdhook.detach().cpu() - official.detach().cpu()).abs()
    return {
        "atol": atol,
        "rtol": rtol,
        "max_abs_difference": float(difference.max()),
        "matches": bool(torch.allclose(tdhook.detach().cpu(), official.detach().cpu(), atol=atol, rtol=rtol)),
    }
