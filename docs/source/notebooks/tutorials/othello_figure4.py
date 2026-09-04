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
from torch.nn import functional as F

from tdhook.session import HookSession
from tdhook.targets import Target

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
    with torch.inference_mode(), HookSession(model) as session:
        captured = session.capture(_target(layer))
        logits = model(inputs)[0]
    return captured.values[-1], logits


def _replace(model: torch.nn.Module, inputs: torch.Tensor, layer: int, replacement: torch.Tensor) -> torch.Tensor:
    with torch.inference_mode(), HookSession(model) as session:
        session.replace(_target(layer), replacement)
        return model(inputs)[0]


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
    option_1_activation, option_1_logits = _capture(model, option_1, layer)
    option_2_activation, option_2_logits = _capture(model, option_2, layer)
    desired = _probe_logits(option_1_activation[:, -1], probe).argmax(-1).expand(len(option_2), -1)
    optimized, _, _ = _inverse_map(option_2_activation[:, -1], desired, probe)
    delta = optimized - option_2_activation[:, -1]

    replacement = option_2_activation.clone()
    replacement[:, -1] = optimized
    intervention_logits = _replace(model, option_2, layer, replacement)

    sham_logits = _replace(model, option_2, layer, option_2_activation)

    wrong_layer = (layer + 4) % 8
    wrong_activation, _ = _capture(model, option_2, wrong_layer)
    wrong_replacement = wrong_activation.clone()
    wrong_replacement[:, -1] += delta
    wrong_logits = _replace(model, option_2, wrong_layer, wrong_replacement)

    random_desired = _probe_logits(option_1_activation[:, -1], randomized_probe).argmax(-1).expand(len(option_2), -1)
    random_optimized, _, _ = _inverse_map(option_2_activation[:, -1], random_desired, randomized_probe)
    random_replacement = option_2_activation.clone()
    random_replacement[:, -1] = random_optimized
    randomized_logits = _replace(model, option_2, layer, random_replacement)

    reference = option_1_logits[:, -1].expand(len(option_2), -1)
    clean_similarity = _cosine(option_2_logits[:, -1], reference).mean()
    scores = {
        "clean": float(clean_similarity.cpu()),
        "intervention": float(_cosine(intervention_logits[:, -1], reference).mean().cpu()),
        "sham": float(_cosine(sham_logits[:, -1], reference).mean().cpu()),
        "wrong_layer": float(_cosine(wrong_logits[:, -1], reference).mean().cpu()),
        "randomized_probe": float(_cosine(randomized_logits[:, -1], reference).mean().cpu()),
    }
    return scores


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

    captures = []
    with torch.inference_mode(), HookSession(model) as session:
        for layer in range(8):
            captures.append(session.capture(_target(layer)))
        logits = model(inputs)[0]
    layer_accuracy = []
    for captured, probe in zip(captures, probes, strict=True):
        predictions = _probe_logits(captured.values[-1], probe).argmax(-1)
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
    summaries = results["summaries"]
    behavior = results["behavior_and_probe"]
    layers = results["layers"]
    game_lengths = results["game_lengths"]
    colors = {"blue": "#2563eb", "orange": "#f97316", "green": "#0f766e", "gray": "#94a3b8"}

    with plt.style.context("seaborn-v0_8-whitegrid"):
        figure, axes = plt.subplots(1, 3, figsize=(15, 4.3), constrained_layout=True)

        accuracy = 100 * np.asarray(behavior["layer_probe_accuracy"])
        axes[0].plot(layers, accuracy, marker="o", linewidth=2.5, color=colors["blue"])
        axes[0].axhline(
            100 * behavior["paper_targets"]["deep_layer_probe_accuracy"],
            color=colors["orange"],
            linestyle="--",
            label="paper: deep layers",
        )
        axes[0].set(title="The board becomes linearly readable", xlabel="Layer", ylabel="Probe accuracy (%)")
        axes[0].set_xticks(layers)
        axes[0].legend(frameon=False)

        gain = np.asarray(summaries["intervention_gain"]["mean"], dtype=float)
        image = axes[1].imshow(gain, aspect="auto", origin="lower", cmap="magma")
        axes[1].set(
            title="Editing the board state changes move logits",
            xlabel="Moves played",
            ylabel="Layer",
        )
        axes[1].set_xticks(range(len(game_lengths)), game_lengths)
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
        axes[2].set(title="Only the meaningful edit has a large effect", xlabel="Early-game gain, layers 3-5")
        axes[2].invert_yaxis()

        for axis in axes:
            axis.spines[["top", "right"]].set_visible(False)
    return figure
