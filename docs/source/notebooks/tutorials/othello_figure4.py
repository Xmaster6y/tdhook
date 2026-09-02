"""Executable Figure 4 reproduction helpers for the Othello research notebook.

This module lives beside the notebook so the resource-intensive scientific
workflow remains separate from TDHook's public library API.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
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
EXPECTED_SHA256 = {
    MODEL_SOURCE_NAME: "f3137b3f959bb7da58e2f53cd10095ac75e76ee6235dd1300659fab1e1a67b67",
    MODEL_NAME: "7702e7072200a4f7758b0c1c09d835dff7fd8785082a07966543c012871de8ba",
    PROBE_TEMPLATE.format(layer=1): "c4f1b3522f4b0cd77f5b68013099a19a522350e49f6fb998303885a748c262ce",
    PROBE_TEMPLATE.format(layer=2): "c9dafc83c773e54ba2140c25af96e234c594ebe8331eb47271c3f217eea561a0",
    PROBE_TEMPLATE.format(layer=3): "61effe04c928a3032178344de7fd7fa0a9404efe59d383f4988da8880159a35d",
    PROBE_TEMPLATE.format(layer=4): "51692249fc2c6c698f4cf878c8349c2049c441a9bfa6a79bfe1d513e7486b8b6",
    PROBE_TEMPLATE.format(layer=5): "fa1df6f2e85d5ee7618781bcfb9c147994e90651e16c541c334b6419cb8c517c",
    PROBE_TEMPLATE.format(layer=6): "6ff9fd51c7bffa85cb940aad0c379dcb587d3bcf102fdb5652704f8a92c92582",
    PROBE_TEMPLATE.format(layer=7): "8b3d2b8b572038520d841d74bb5b3f3ad770ba711e0b524c313573f71577ba3c",
    PROBE_TEMPLATE.format(layer=8): "f4f9bcaccb261e98b49454ef6e3250a01b18fcbc75967392efdba0fc43b98257",
}
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download_assets(cache: Path) -> dict[str, Path]:
    cache.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for name, url in ASSET_URLS.items():
        path = cache / name
        if not path.exists():
            urllib.request.urlretrieve(url, path)
        observed = _sha256(path)
        if observed != EXPECTED_SHA256[name]:
            raise RuntimeError(f"{name} checksum mismatch: {observed}")
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
    return captured.value, logits


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
) -> tuple[dict[str, float], dict[str, float]]:
    option_1_activation, option_1_logits = _capture(model, option_1, layer)
    option_2_activation, option_2_logits = _capture(model, option_2, layer)
    desired = _probe_logits(option_1_activation[:, -1], probe).argmax(-1).expand(len(option_2), -1)
    optimized, steps, loss = _inverse_map(option_2_activation[:, -1], desired, probe)
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
    random_optimized, random_steps, random_loss = _inverse_map(
        option_2_activation[:, -1], random_desired, randomized_probe
    )
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
    diagnostics = {
        "inverse_steps": float(steps),
        "inverse_loss": loss,
        "randomized_inverse_steps": float(random_steps),
        "randomized_inverse_loss": random_loss,
        "sham_max_abs_logit_error": float((sham_logits - option_2_logits).abs().max().cpu()),
    }
    return scores, diagnostics


def _bootstrap_mean_ci(values: np.ndarray, *, seed: int, samples: int = 2000) -> tuple[np.ndarray, np.ndarray]:
    generator = np.random.default_rng(seed)
    draws = generator.integers(0, values.shape[0], size=(samples, values.shape[0]))
    bootstrapped = np.nanmean(values[draws], axis=1)
    return np.nanquantile(bootstrapped, 0.025, axis=0), np.nanquantile(bootstrapped, 0.975, axis=0)


def _json_array(value: np.ndarray) -> list[Any]:
    return np.where(np.isfinite(value), value, None).tolist()


def _reference_parity(
    model: torch.nn.Module,
    option_1: torch.Tensor,
    option_2: torch.Tensor,
    layer: int,
    probe: tuple[torch.Tensor, torch.Tensor],
) -> float:
    option_1_activation, _ = _capture(model, option_1, layer)
    option_2_activation, _ = _capture(model, option_2[:1], layer)
    desired = _probe_logits(option_1_activation[:, -1], probe).argmax(-1)
    optimized, _, _ = _inverse_map(option_2_activation[:, -1], desired, probe)
    replacement = option_2_activation.clone()
    replacement[:, -1] = optimized
    tdhook_logits = _replace(model, option_2[:1], layer, replacement)

    with torch.inference_mode():
        staged = model.forward_1st_stage(layer + 1, option_2[:1])
        staged[:, -1] = optimized
        downstream = model.forward_2nd_stage(staged, layer + 1, -1)[-1]
        reference_logits = model.predict(downstream)[0]
    return float((tdhook_logits - reference_logits).abs().max().cpu())


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
        predictions = _probe_logits(captured.value, probe).argmax(-1)
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
        "gates": {
            "deep_layer_probe_accuracy": abs(deep_accuracy - 0.995) <= tolerance,
            "legal_move_rate": abs(legal_rate - 0.999) <= tolerance,
        },
    }


def run_figure4_reproduction(
    *,
    cache: Path,
    games_int: np.ndarray,
    games_string: np.ndarray,
    othello: ModuleType,
    output_path: Path,
    number_games: int = 50,
    seed: int = 44,
) -> dict[str, Any]:
    """Execute the preregistered Figure 4 sweep and write its JSON artifact."""
    if number_games != 50:
        raise ValueError("scientific mode requires exactly 50 games")
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
    inverse_steps = np.full((number_games, 8, len(game_lengths)), np.nan)
    randomized_inverse_steps = np.full_like(inverse_steps, np.nan)
    sham_max_abs_error = 0.0
    playable = [square for square in range(64) if square not in (27, 28, 35, 36)]
    square_to_token = {square: token + 1 for token, square in enumerate(playable)}

    parity_case = None
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
                if parity_case is None and layer == 3 and game_length == 10:
                    parity_case = (option_1, option_2, layer, probe)
                scores, diagnostics = _evaluate_options(model, option_1, option_2, layer, probe, random_probe)
                for name, score in scores.items():
                    per_game[name][sample_index, layer, length_index] = score
                inverse_steps[sample_index, layer, length_index] = diagnostics["inverse_steps"]
                randomized_inverse_steps[sample_index, layer, length_index] = diagnostics["randomized_inverse_steps"]
                sham_max_abs_error = max(sham_max_abs_error, diagnostics["sham_max_abs_logit_error"])

    if parity_case is None:
        raise RuntimeError("the fixed parity case was not evaluated")
    parity_max_abs_difference = _reference_parity(model, *parity_case)

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
    gates = {
        **behavior["gates"],
        "tdhook_reference_parity": parity_max_abs_difference <= 1e-5,
        "sham_identity": sham_max_abs_error <= 1e-6,
        "middle_over_late_ordering": float(ordering_lower[0]) > 0,
    }

    artifact = {
        "schema_version": 1,
        "claim": "50-game Figure 4 ordering reproduction",
        "provenance": {
            "hazineh_revision": HAZINEH_REVISION,
            "asset_sha256": {name: _sha256(path) for name, path in paths.items()},
            "torch": torch.__version__,
            "device": str(device),
        },
        "protocol": {
            "seed": seed,
            "number_games": number_games,
            "game_indices": game_indices.tolist(),
            "game_lengths": game_lengths.tolist(),
            "layers": list(range(1, 9)),
            "inverse_optimizer": {"name": "adam", "learning_rate": 0.05, "max_steps": 3000},
            "bootstrap_samples": 2000,
            "controls": ["sham", "wrong_layer", "randomized_probe"],
        },
        "behavior_and_probe": behavior,
        "tdhook_reference_parity_max_abs_difference": parity_max_abs_difference,
        "sham_max_abs_logit_error": sham_max_abs_error,
        "ordering": {
            "early_game_max_length": 20,
            "middle_layers": [3, 4, 5],
            "late_layers": [7, 8],
            "mean_paired_gain_difference": float(np.nanmean(ordering_difference)),
            "bootstrap_95_ci": [float(ordering_lower[0]), float(ordering_upper[0])],
        },
        "summaries": summaries,
        "diagnostics": {
            "inverse_steps_mean": float(np.nanmean(inverse_steps)),
            "inverse_steps_max": float(np.nanmax(inverse_steps)),
            "randomized_inverse_steps_mean": float(np.nanmean(randomized_inverse_steps)),
            "randomized_inverse_steps_max": float(np.nanmax(randomized_inverse_steps)),
        },
        "gates": gates,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2, allow_nan=False) + "\n")
    if not all(gates.values()):
        raise AssertionError(f"reproduction gates failed: {gates}")
    return artifact


def plot_figure4_reproduction(artifact: dict[str, Any]) -> plt.Figure:
    """Display the causal result, controls, and uncertainty on aligned axes."""
    summaries = artifact["summaries"]
    panels = (
        ("clean", "Option 1 vs option 2 (clean)"),
        ("intervention", "Option 1 vs option 2 (intervention)"),
        ("intervention_gain", "Causal similarity gain"),
        ("wrong_layer_gain", "Wrong-layer control gain"),
        ("randomized_probe_gain", "Randomized-probe control gain"),
    )
    intervention = summaries["intervention_gain"]
    uncertainty = (
        np.asarray(intervention["bootstrap_95_ci_upper"], dtype=float)
        - np.asarray(intervention["bootstrap_95_ci_lower"], dtype=float)
    ) / 2
    figure, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    for axis, (name, title) in zip(axes.flat, panels, strict=False):
        image = axis.imshow(np.asarray(summaries[name]["mean"], dtype=float), aspect="auto", origin="lower")
        axis.set_title(title)
        figure.colorbar(image, ax=axis, shrink=0.8)
    image = axes.flat[-1].imshow(uncertainty, aspect="auto", origin="lower")
    axes.flat[-1].set_title("Causal gain bootstrap 95% CI half-width")
    figure.colorbar(image, ax=axes.flat[-1], shrink=0.8)
    for axis in axes.flat:
        axis.set_xlabel("Moves played (5-move steps)")
        axis.set_ylabel("Probe layer")
    return figure
