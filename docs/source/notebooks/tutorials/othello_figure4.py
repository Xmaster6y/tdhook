"""Executable Figure 4 reproduction helpers for the Othello research notebook.

This module lives beside the notebook so the resource-intensive scientific
workflow remains separate from TDHook's public library API.
"""

from __future__ import annotations

import importlib.util
import math
import urllib.request
from pathlib import Path
from types import ModuleType
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from tensordict import NonTensorData, TensorDict
from tensordict.nn import TensorDictModule
from torch.nn import functional as F

from tdhook.latent import ActivationCaching, SteeringVectors
from tdhook.targets import Target
from tdhook.workflow import Workflow

HAZINEH_REVISION = "e52217b4b756d22579b28f24c7b1b2355c8f8914"
HAZINEH_BASE = (
    f"https://raw.githubusercontent.com/DeanHazineh/Emergent-World-Representations-Othello/{HAZINEH_REVISION}"
)
MODEL_NAME = "hazineh-8l8h-model.ckpt"
MODEL_SOURCE_NAME = "hazineh-model.py"
PROBE_TEMPLATE = "hazineh-8l8h-probe-layer-{layer}.ckpt"
ASSET_URLS = {
    MODEL_SOURCE_NAME: f"{HAZINEH_BASE}/EWOthello/mingpt/model.py",
    MODEL_NAME: (f"{HAZINEH_BASE}/EWOthello/ckpts/DeanKLi_GPT_Synthetic_8L8H/GPT_Synthetic_8Layers_8Heads.ckpt"),
    **{
        PROBE_TEMPLATE.format(layer=layer): (
            f"{HAZINEH_BASE}/EWOthello/ckpts/DeanKLi_GPT_Synthetic_8L8H/linearProbe_Map_New_8L8H_GPT_Layer{layer}.ckpt"
        )
        for layer in range(1, 9)
    },
}


def _download_assets(cache: Path) -> dict[str, Path]:
    cache.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for name, url in ASSET_URLS.items():
        path = cache / name
        if not path.exists():
            urllib.request.urlretrieve(url, path)
        paths[name] = path
    return paths


def _load_released_model(paths: dict[str, Path], device: torch.device) -> torch.nn.Module:
    spec = importlib.util.spec_from_file_location("released_hazineh_model", paths[MODEL_SOURCE_NAME])
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load the pinned Hazineh model implementation")
    released = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(released)
    config = released.GPTConfig(vocab_size=61, block_size=59, n_layer=8, n_head=8, n_embd=512)
    model = released.GPTforProbeIA_ModV1(config)
    state = torch.load(paths[MODEL_NAME], map_location=device, weights_only=False)
    model.load_state_dict(state)
    model.eval().to(device)
    return model


def _load_probes(paths: dict[str, Path], device: torch.device) -> list[tuple[torch.Tensor, torch.Tensor]]:
    probes = []
    for layer in range(1, 9):
        state = torch.load(paths[PROBE_TEMPLATE.format(layer=layer)], map_location=device, weights_only=False)
        probes.append((state["proj.weight"], state["proj.bias"]))
    return probes


def _target(layer: int) -> Target:
    return Target(f"blocks.{layer}", "activation", -1, tuple(range(512)))


def _capture(model: torch.nn.Module, inputs: torch.Tensor, layer: int) -> tuple[torch.Tensor, torch.Tensor]:
    target = _target(layer)
    cache_key = ("activations", "selected")
    artifacts = TensorDict({"input": inputs}, batch_size=[])
    model_module = TensorDictModule(model, in_keys=["input"], out_keys=["output", "aux"])
    with torch.inference_mode():
        result = Workflow(ActivationCaching(target, cache_key=cache_key))(model_module, artifacts)
    return result[(*cache_key, target.module_path)], result["output"]


def _replace(model: torch.nn.Module, inputs: torch.Tensor, layer: int, replacement: torch.Tensor) -> torch.Tensor:
    method = SteeringVectors([_target(layer)], lambda **_kwargs: replacement)
    model_module = TensorDictModule(model, in_keys=["input"], out_keys=["output", "aux"])
    with torch.inference_mode():
        result = Workflow(method)(model_module, TensorDict({"input": inputs}, batch_size=[]))
    return result["output"]


def _probe_logits(value: torch.Tensor, probe: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    weight, bias = probe
    return F.linear(value, weight, bias).reshape(*value.shape[:-1], 64, 3)


def _inverse_map(
    initial: torch.Tensor,
    desired_board: torch.Tensor,
    probe: tuple[torch.Tensor, torch.Tensor],
    *,
    max_steps: int = 3000,
    learning_rate: float = 5e-2,
) -> tuple[torch.Tensor, int, float]:
    """Reproduce the authors' Adam inverse map, batched over move options."""
    value = initial.detach().clone().requires_grad_(True)
    optimizer = torch.optim.Adam([value], lr=learning_rate)
    final_loss = math.inf
    for step in range(max_steps):
        optimizer.zero_grad()
        logits = _probe_logits(value, probe)
        loss = F.cross_entropy(logits.flatten(0, -2), desired_board.flatten())
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            updated_logits = _probe_logits(value, probe)
            updated_loss = F.cross_entropy(updated_logits.flatten(0, -2), desired_board.flatten())
            final_loss = float(updated_loss.cpu())
            if step % 10 == 0 and torch.equal(updated_logits.argmax(-1), desired_board):
                return value.detach(), step + 1, final_loss
    raise RuntimeError(f"inverse map did not converge within {max_steps} optimizer steps (loss={final_loss:.6g})")


def _cosine(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    # The reference excludes token 0 (pass) and inserts four zero-valued centre squares.
    # Inserting zeros leaves cosine similarity unchanged, so the 60 playable logits suffice.
    return F.cosine_similarity(left[..., 1:], right[..., 1:], dim=-1)


def _option_inputs(
    game_tokens: np.ndarray,
    game_squares: np.ndarray,
    number_played_moves: int,
    othello: ModuleType,
    square_to_token: dict[int, int],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    board = othello.OthelloBoardState()
    for move in game_squares[:number_played_moves].tolist():
        board.umpire(int(move))
    valid_moves = board.get_valid_moves()
    if len(valid_moves) < 2:
        return None
    prefix = game_tokens[:number_played_moves].tolist()
    option_1 = torch.tensor([prefix + [square_to_token[int(valid_moves[0])]]], device=device)
    option_2 = torch.tensor([prefix + [square_to_token[int(move)]] for move in valid_moves[1:]], device=device)
    return option_1, option_2


def _randomized_probe(
    probe: tuple[torch.Tensor, torch.Tensor], layer: int, seed: int
) -> tuple[torch.Tensor, torch.Tensor]:
    weight, bias = probe
    generator = torch.Generator(device="cpu").manual_seed(seed + layer)
    permutation = torch.randperm(weight.shape[-1], generator=generator).to(weight.device)
    return weight[:, permutation], bias


def _evaluate_options(
    model: torch.nn.Module,
    option_1: torch.Tensor,
    option_2: torch.Tensor,
    layer: int,
    probe: tuple[torch.Tensor, torch.Tensor],
    randomized_probe: tuple[torch.Tensor, torch.Tensor],
) -> dict[str, float]:
    workflow = option_intervention_workflow(model, layer, probe, randomized_probe)
    result = workflow(model, TensorDict({"reference_input": option_1, "alternative_input": option_2}, []))
    return result["metrics", "scores"]


def option_intervention_workflow(model, layer, probe, randomized_probe) -> Workflow:
    """Capture both options, derive replacements, apply controls, and score.

    Model and probe configuration is fixed when constructing the workflow.
    Every per-game activation, replacement, and prediction travels through
    declared TensorDict keys rather than a closure over an earlier result.
    """

    def replacements(reference, alternative):
        desired = _probe_logits(reference[:, -1], probe).argmax(-1).expand(len(alternative), -1)
        optimized, _, _ = _inverse_map(alternative[:, -1], desired, probe)
        replacement = alternative.clone()
        replacement[:, -1] = optimized
        delta = optimized - alternative[:, -1]
        random_desired = _probe_logits(reference[:, -1], randomized_probe).argmax(-1).expand(len(alternative), -1)
        random_optimized, _, _ = _inverse_map(alternative[:, -1], random_desired, randomized_probe)
        random_replacement = alternative.clone()
        random_replacement[:, -1] = random_optimized
        return replacement, delta, random_replacement

    def wrong_layer_prediction(inputs, delta):
        wrong_layer = (layer + 4) % 8
        wrong_activation, _ = _capture(model, inputs, wrong_layer)
        replacement = wrong_activation.clone()
        replacement[:, -1] += delta
        return _replace(model, inputs, wrong_layer, replacement)

    def score(reference, clean, intervention, sham, wrong_layer, randomized):
        reference = reference[:, -1].expand(len(clean), -1)
        predictions = dict(
            clean=clean, intervention=intervention, sham=sham, wrong_layer=wrong_layer, randomized_probe=randomized
        )
        return NonTensorData(
            {name: float(_cosine(logits[:, -1], reference).mean().cpu()) for name, logits in predictions.items()}
        )

    steps = [
        TensorDictModule(
            lambda inputs: _capture(model, inputs, layer),
            ["reference_input"],
            [("activations", "reference"), ("logits", "reference")],
        ),
        TensorDictModule(
            lambda inputs: _capture(model, inputs, layer),
            ["alternative_input"],
            [("activations", "alternative"), ("logits", "clean")],
        ),
        TensorDictModule(
            replacements,
            [("activations", "reference"), ("activations", "alternative")],
            [("replacement", "intervention"), ("replacement", "delta"), ("replacement", "randomized")],
        ),
    ]
    for condition, source in (
        ("intervention", ("replacement", "intervention")),
        ("sham", ("activations", "alternative")),
        ("randomized", ("replacement", "randomized")),
    ):
        steps.append(
            TensorDictModule(
                lambda inputs, value: _replace(model, inputs, layer, value),
                ["alternative_input", source],
                [("logits", condition)],
            )
        )
    steps.extend(
        [
            TensorDictModule(
                wrong_layer_prediction, ["alternative_input", ("replacement", "delta")], [("logits", "wrong_layer")]
            ),
            TensorDictModule(
                score,
                [
                    ("logits", name)
                    for name in ("reference", "clean", "intervention", "sham", "wrong_layer", "randomized")
                ],
                [("metrics", "scores")],
            ),
        ]
    )
    return Workflow(*steps)


def prepare_behavior_example(cache, games_int, games_string, othello, *, game_index=0, game_length=10, layer=3):
    """Fix one example before observing effects; use the first two legal moves."""
    paths = _download_assets(cache)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _load_released_model(paths, device)
    probe = _load_probes(paths, device)[layer]
    playable = [square for square in range(64) if square not in (27, 28, 35, 36)]
    options = _option_inputs(
        games_int[game_index],
        games_string[game_index],
        game_length,
        othello,
        {square: token + 1 for token, square in enumerate(playable)},
        device,
    )
    if options is None:
        raise ValueError("The configured position needs at least two legal moves")
    first, alternatives = options
    data = TensorDict({"reference_input": first, "alternative_input": alternatives[:1]}, [])
    return model, probe, data


def prediction_view(reference, clean, intervention, sham, scores):
    """Materialize plot-ready values; preserve pass probability and signed effects."""
    logits = torch.stack([value[0, -1].detach().float().cpu() for value in (reference, clean, intervention)])
    probabilities = logits.softmax(-1)
    return NonTensorData(
        {
            "probabilities": probabilities.tolist(),
            "scores": dict(scores),
            "sham_max_abs_logit_difference": float((clean - sham).abs().max().cpu()),
        }
    )


def plot_prediction_comparison(view):
    """Render exported values only: no model, probe, or optimization access."""
    from notebook_figures import STYLE

    probabilities = np.asarray(view["probabilities"])
    if probabilities.shape != (3, 61) or not np.isfinite(probabilities).all():
        raise ValueError("Expected three finite 61-token probability distributions")
    if (probabilities < 0).any() or not np.allclose(probabilities.sum(-1), 1):
        raise ValueError("Each probability distribution must be nonnegative and sum to one")
    playable = [square for square in range(64) if square not in (27, 28, 35, 36)]
    boards = np.full((3, 64), np.nan)
    boards[:, playable] = probabilities[:, 1:]
    with plt.rc_context(STYLE):
        figure, axes = plt.subplots(1, 3, figsize=(13, 4.5), constrained_layout=True)
        for index, (axis, title) in enumerate(
            zip(axes, ("Move A · reference", "Move B · unchanged", "Move B · intervened"))
        ):
            image = axis.imshow(
                boards[index].reshape(8, 8), vmin=0, vmax=float(probabilities[:, 1:].max()), cmap="Blues"
            )
            axis.set(
                title=title, xticks=range(8), xticklabels=list("ABCDEFGH"), yticks=range(8), yticklabels=range(1, 9)
            )
            distance = float(np.abs(probabilities[index] - probabilities[0]).sum() / 2)
            axis.set_xlabel(f"Pass: {probabilities[index, 0]:.3f}\nProbability distance to A: {distance:.3f}")
        figure.colorbar(image, ax=axes, label="Next-move probability", shrink=0.8)
    return figure


def _bootstrap_mean_ci(values: np.ndarray, *, seed: int, samples: int = 2000) -> tuple[np.ndarray, np.ndarray]:
    generator = np.random.default_rng(seed)
    draws = generator.integers(0, values.shape[0], size=(samples, values.shape[0]))
    bootstrapped = np.nanmean(values[draws], axis=1)
    return np.nanquantile(bootstrapped, 0.025, axis=0), np.nanquantile(bootstrapped, 0.975, axis=0)


def _json_array(value: np.ndarray) -> list[Any]:
    return np.where(np.isfinite(value), value, None).tolist()


def _probe_and_behavior_metrics(
    model: torch.nn.Module,
    probes: list[tuple[torch.Tensor, torch.Tensor]],
    games_int: np.ndarray,
    games_string: np.ndarray,
    othello: ModuleType,
    device: torch.device,
) -> dict[str, Any]:
    inputs = torch.from_numpy(games_int[:100, :59].astype(np.int64)).to(device)
    states = []
    for sequence in games_string[:100, :59]:
        board = othello.OthelloBoardState()
        sequence_states = []
        for move in sequence.tolist():
            board.umpire(int(move))
            sequence_states.append(board.state.copy())
        states.append(np.stack(sequence_states))
    states_tensor = torch.from_numpy(np.stack(states)).to(device)
    parity = torch.tensor([(-1) ** position for position in range(59)], device=device).view(1, 59, 1, 1)
    labels = (states_tensor * parity + 1).long().reshape(100, 59, 64)

    cache_key = ("activations", "layers")
    model_module = TensorDictModule(model, in_keys=["input"], out_keys=["output", "aux"])
    with torch.inference_mode():
        result = Workflow(
            ActivationCaching(r"module.blocks\.[0-7]$", cache_key=cache_key),
        )(model_module, TensorDict({"input": inputs}, batch_size=[]))
    logits = result["output"]
    layer_accuracy = []
    for layer, probe in enumerate(probes):
        predictions = _probe_logits(result[(*cache_key, f"module.blocks.{layer}")], probe).argmax(-1)
        layer_accuracy.append(float((predictions[:, 5:54] == labels[:, 5:54]).float().mean().cpu()))

    playable = [square for square in range(64) if square not in (27, 28, 35, 36)]
    illegal = total = 0
    predicted_tokens = logits.argmax(-1).cpu().numpy()
    for game_index, sequence in enumerate(games_string[:100, :59]):
        board = othello.OthelloBoardState()
        for position, move in enumerate(sequence.tolist()):
            board.umpire(int(move))
            if position < 58:
                token = int(predicted_tokens[game_index, position])
                predicted_square = -1 if token == 0 else playable[token - 1]
                illegal += int(predicted_square not in board.get_valid_moves())
                total += 1
    legal_rate = 1 - illegal / total
    deep_accuracy = float(np.mean(layer_accuracy[5:8]))
    tolerance = 0.005
    return {
        "layer_probe_accuracy": layer_accuracy,
        "deep_layer_6_to_8_probe_accuracy": deep_accuracy,
        "legal_move_rate": legal_rate,
        "evaluated_next_move_positions": total,
        "paper_targets": {"deep_layer_probe_accuracy": 0.995, "legal_move_rate": 0.999},
        "absolute_tolerance": tolerance,
    }


def run_figure4_reproduction(
    *,
    cache: Path,
    games_int: np.ndarray,
    games_string: np.ndarray,
    othello: ModuleType,
    number_games: int = 50,
    seed: int = 44,
) -> dict[str, Any]:
    """Run the Figure 4 sweep and write its measurements to JSON."""
    if not 1 <= number_games <= 100:
        raise ValueError("number_games must be between 1 and 100")
    if len(games_int) < 100 or len(games_string) < 100:
        raise ValueError("the released-data validation population requires at least 100 games")
    games_int = games_int[:100, :59]
    games_string = games_string[:100, :59]
    paths = _download_assets(cache)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _load_released_model(paths, device)
    probes = _load_probes(paths, device)
    behavior = _probe_and_behavior_metrics(model, probes, games_int, games_string, othello, device)

    game_lengths = np.arange(0, 58, 5)
    game_indices = np.random.default_rng(seed).choice(len(games_int), number_games, replace=False)
    condition_names = ("clean", "intervention", "sham", "wrong_layer", "randomized_probe")
    per_game = {name: np.full((number_games, 8, len(game_lengths)), np.nan) for name in condition_names}
    playable = [square for square in range(64) if square not in (27, 28, 35, 36)]
    square_to_token = {square: token + 1 for token, square in enumerate(playable)}

    for sample_index, game_index in enumerate(game_indices):
        for layer, probe in enumerate(probes):
            random_probe = _randomized_probe(probe, layer, seed)
            for length_index, game_length in enumerate(game_lengths):
                options = _option_inputs(
                    games_int[game_index],
                    games_string[game_index],
                    int(game_length),
                    othello,
                    square_to_token,
                    device,
                )
                if options is None:
                    continue
                option_1, option_2 = options
                scores = _evaluate_options(model, option_1, option_2, layer, probe, random_probe)
                for name, score in scores.items():
                    per_game[name][sample_index, layer, length_index] = score

    gains = {
        name: per_game[name] - per_game["clean"]
        for name in ("intervention", "sham", "wrong_layer", "randomized_probe")
    }
    summaries: dict[str, Any] = {}
    for name, values in {**per_game, **{f"{key}_gain": value for key, value in gains.items()}}.items():
        lower, upper = _bootstrap_mean_ci(values, seed=seed)
        summaries[name] = {
            "mean": _json_array(np.nanmean(values, axis=0)),
            "bootstrap_95_ci_lower": _json_array(lower),
            "bootstrap_95_ci_upper": _json_array(upper),
        }

    early = game_lengths <= 20
    middle_per_game = np.nanmean(gains["intervention"][:, 2:5, :][:, :, early], axis=(1, 2))
    late_per_game = np.nanmean(gains["intervention"][:, 6:8, :][:, :, early], axis=(1, 2))
    ordering_difference = middle_per_game - late_per_game
    ordering_lower, ordering_upper = _bootstrap_mean_ci(ordering_difference[:, None], seed=seed)
    results = {
        "number_games": number_games,
        "game_lengths": game_lengths.tolist(),
        "layers": list(range(1, 9)),
        "behavior_and_probe": behavior,
        "middle_over_late": {
            "mean": float(np.nanmean(ordering_difference)),
            "interval": [float(ordering_lower[0]), float(ordering_upper[0])],
        },
        "summaries": summaries,
    }
    return results


def plot_figure4_reproduction(results: dict[str, Any]) -> plt.Figure:
    """Plot the main behavioral and intervention results."""
    from matplotlib.colors import TwoSlopeNorm

    from notebook_figures import STYLE

    summaries = results["summaries"]
    behavior = results["behavior_and_probe"]
    layers = results["layers"]
    game_lengths = results["game_lengths"]
    colors = {"blue": "#2563eb", "orange": "#f97316", "green": "#0f766e", "gray": "#94a3b8"}

    with plt.rc_context(STYLE):
        figure, axes = plt.subplots(1, 3, figsize=(17, 5.5), constrained_layout=True)
        figure.suptitle("Othello interventions · 50 games, eight layers", fontsize=21)

        accuracy = 100 * np.asarray(behavior["layer_probe_accuracy"])
        axes[0].plot(layers, accuracy, marker="o", linewidth=2.5, color=colors["blue"])
        axes[0].axhline(
            100 * behavior["paper_targets"]["deep_layer_probe_accuracy"],
            color=colors["orange"],
            linestyle="--",
            label="paper: deep layers",
        )
        axes[0].set(title="Board decoding", xlabel="Layer", ylabel="Probe accuracy (%)")
        axes[0].set_xticks(layers)
        axes[0].legend(frameon=False)

        gain = np.asarray(summaries["intervention_gain"]["mean"], dtype=float)
        limit = max(float(np.nanmax(np.abs(gain))), 1e-6)
        norm = TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)
        image = axes[1].imshow(gain, aspect="auto", origin="lower", cmap="RdBu_r", norm=norm)
        axes[1].set(
            title="Intervention effect",
            xlabel="Moves played",
            ylabel="Layer",
        )
        axes[1].set_xticks(range(0, len(game_lengths), 2), game_lengths[::2])
        axes[1].set_yticks(range(len(layers)), layers)
        figure.colorbar(image, ax=axes[1], label="Cosine-similarity gain", shrink=0.82)

        early = np.asarray(game_lengths) <= 20
        conditions = (
            ("intervention_gain", "Board edit", colors["blue"]),
            ("wrong_layer_gain", "Wrong layer", colors["orange"]),
            ("randomized_probe_gain", "Random probe", colors["green"]),
            ("sham_gain", "Sham", colors["gray"]),
        )
        values = [np.nanmean(np.asarray(summaries[name]["mean"])[2:5, :][:, early]) for name, _, _ in conditions]
        bars = axes[2].barh(
            [label for _, label, _ in conditions],
            values,
            color=[color for _, _, color in conditions],
        )
        axes[2].bar_label(bars, fmt="%.3f", padding=4)
        axes[2].axvline(0, color="#334155", linewidth=0.8)
        axes[2].set(title="Controls", xlabel="Early-game gain, layers 3-5")
        axes[2].invert_yaxis()
        axes[2].margins(x=0.3)

        for axis in axes:
            axis.spines[["top", "right"]].set_visible(False)
    return figure
