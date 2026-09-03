"""Deterministic helpers for the ROME causal-tracing reproduction notebook.

This module deliberately contains mechanics, not checked-in scientific results.  The
resource-intensive notebook writes the per-case evidence used to decide its gates.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from tdhook.session import HookSession
from tdhook.targets import Target


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

    patches: dict[str, list[int]] = defaultdict(list)
    for token, module_path in states_to_patch:
        patches[module_path].append(token)

    embedding = Target(embedding_path, "activation", 1, tuple(range(start, stop)))
    with torch.no_grad(), HookSession(model) as session:
        clean_embeddings = session.capture(embedding)
        session.replace(
            embedding,
            clean_embeddings,
            transform=lambda value: _corrupt_subject(
                value,
                seed=config.noise_seed,
                noise_level=config.noise_level,
                replace=config.replace_noise,
            ),
        )
        for module_path, tokens in patches.items():
            state = Target(
                module_path,
                "activation",
                1,
                tuple(sorted(set(tokens))),
                output_path=(state_output_paths or {}).get(module_path, ()),
            )
            clean_state = session.capture(state)
            session.replace(state, clean_state, transform=_restore_clean)
        output = model(**model_inputs)

    probabilities = torch.softmax(_logits(output)[1:, -1, :], dim=-1)
    return probabilities[:, answer_token].mean()


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
) -> tuple[Tensor, dict[str, int]]:
    """Restore each token/layer state and return scores plus an exact pass budget."""

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
    return torch.stack(rows), {
        "model_pass_budget": len(token_indices) * len(layer_paths),
        "model_passes": len(token_indices) * len(layer_paths),
    }


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
) -> tuple[Tensor, dict[str, int]]:
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
    passes = len(token_indices) * len(layer_paths)
    return torch.stack(rows), {"model_pass_budget": passes, "model_passes": passes}


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


def bootstrap_mean_interval(values: Sequence[float], *, seed: int, samples: int = 10_000) -> tuple[float, float]:
    """Return a deterministic percentile 95% confidence interval for a mean."""

    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or not len(array):
        raise ValueError("values must be a non-empty one-dimensional sequence")
    random = np.random.default_rng(seed)
    means = random.choice(array, (samples, len(array)), replace=True).mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def summarize_counterfact(cases: Sequence[Mapping[str, Any]], *, seed: int) -> dict[str, Any]:
    """Aggregate official per-case output into declared metrics and confidence intervals."""

    scores = [case_score(case["post"]) for case in cases]
    summary = {}
    for metric in ("rewrite_efficacy", "paraphrase_generalization", "neighborhood_specificity"):
        values = [score[metric] for score in scores]
        summary[metric] = {
            "mean": float(np.mean(values)),
            "bootstrap_95_ci": bootstrap_mean_interval(values, seed=seed),
        }
    return {"number_cases": len(cases), "metrics": summary}


def parity_report(tdhook: Tensor, official: Tensor, *, atol: float, rtol: float) -> dict[str, Any]:
    """Record, rather than merely assert, fixed-case numerical parity."""

    difference = (tdhook.detach().cpu() - official.detach().cpu()).abs()
    return {
        "atol": atol,
        "rtol": rtol,
        "max_abs_difference": float(difference.max()),
        "matches": bool(torch.allclose(tdhook.detach().cpu(), official.detach().cpu(), atol=atol, rtol=rtol)),
    }


def write_json(path: str | Path, value: Mapping[str, Any]) -> str:
    """Write canonical machine-readable evidence and return its SHA-256."""

    encoded = json.dumps(value, allow_nan=False, indent=2, sort_keys=True).encode() + b"\n"
    Path(path).write_bytes(encoded)
    return sha256(encoded).hexdigest()


def config_dict(config: CausalTraceConfig) -> dict[str, Any]:
    return asdict(config)
