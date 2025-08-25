"""
Script to probe the action of a PPO agent.

Run with:

```
uv run --group scripts -m scripts.torchrl.action_probing
```
"""

import os
import argparse
import re
from collections import defaultdict
from loguru import logger
from tqdm import tqdm
import torch
from torch import nn
from torchrl.envs import TransformedEnv, Compose, DoubleToFloat, StepCounter, set_exploration_type, ExplorationType
from torchrl.envs.libs.gym import GymEnv
from torchrl.collectors import SyncDataCollector
from torchrl.modules import MLP, ProbabilisticActor, NormalParamExtractor, TanhNormal
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from torchrl.data import TensorDictReplayBuffer, SamplerWithoutReplacement, LazyTensorStorage
from tensordict.nn import TensorDictModule
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

from tdhook.acteng.probing import Probing, SklearnProbeManager

# Set plotting style similar to benchmark plots
plt.style.use("default")
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.3
plt.rcParams["axes.facecolor"] = "white"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

total_frames = 1_000_000
frames_per_batch = 1000
sub_batch_size = 100
num_epochs = 20
num_cells = 6
hidden_size = 32


def get_env(args):
    base_env = GymEnv(args.env_name)
    return TransformedEnv(
        base_env,
        Compose(
            DoubleToFloat(),
            StepCounter(),
        ),
    )


def get_replay_buffer():
    return TensorDictReplayBuffer(
        storage=LazyTensorStorage(max_size=total_frames),
        sampler=SamplerWithoutReplacement(),
    )


def get_policy_value(env):
    actor_module = TensorDictModule(
        nn.Sequential(
            MLP(
                in_features=env.observation_spec["observation"].shape[-1],
                out_features=2 * env.action_spec.shape[-1],
                num_cells=[
                    hidden_size,
                ]
                * num_cells,
            ),
            NormalParamExtractor(),
        ),
        in_keys=["observation"],
        out_keys=["loc", "scale"],
    )
    actor = ProbabilisticActor(
        module=actor_module,
        spec=env.action_spec,
        in_keys=["loc", "scale"],
        distribution_class=TanhNormal,
        distribution_kwargs={
            "low": env.action_spec.space.low,
            "high": env.action_spec.space.high,
        },
        return_log_prob=True,
    )
    critic = TensorDictModule(
        MLP(
            in_features=env.observation_spec["observation"].shape[-1],
            out_features=1,
            num_cells=[
                hidden_size,
            ]
            * num_cells,
        ),
        in_keys=["observation"],
        out_keys=["state_value"],
    )
    return actor, critic


def get_loss_advantage(actor, critic):
    advantage_module = GAE(
        gamma=0.99, lmbda=0.95, value_network=critic, average_gae=True, device=device, deactivate_vmap=True
    )

    loss_module = ClipPPOLoss(
        actor_network=actor,
        critic_network=critic,
        clip_epsilon=0.2,
        entropy_bonus=True,
        entropy_coeff=1e-4,
        critic_coeff=1.0,
        loss_critic_type="smooth_l1",
        functional=False,
    )

    optim = torch.optim.Adam(loss_module.parameters(), lr=1e-4)
    return advantage_module, loss_module, optim


def get_collector(env, actor):
    return SyncDataCollector(
        env,
        actor,
        frames_per_batch=frames_per_batch,
        total_frames=total_frames,
    )


def train(env, actor, advantage_module, loss_module, optim, replay_buffer, collector):
    logs = defaultdict(list)
    pbar = tqdm(total=total_frames)

    for i, tensordict_data in enumerate(collector):
        for _ in range(num_epochs):
            advantage_module(tensordict_data)
            data_view = tensordict_data.reshape(-1)
            replay_buffer.extend(data_view.cpu())
            for _ in range(frames_per_batch // sub_batch_size):
                subdata = replay_buffer.sample(sub_batch_size)
                loss_vals = loss_module(subdata.to(device))
                loss_value = loss_vals["loss_objective"] + loss_vals["loss_critic"] + loss_vals["loss_entropy"]

                loss_value.backward()
                torch.nn.utils.clip_grad_norm_(loss_module.parameters(), 1.0)
                optim.step()
                optim.zero_grad()

        logs["reward"].append(tensordict_data["next", "reward"].mean().item())
        pbar.update(tensordict_data.numel())
        cum_reward_str = f"average reward={logs['reward'][-1]: 4.4f} (init={logs['reward'][0]: 4.4f})"
        logs["step_count"].append(tensordict_data["step_count"].max().item())
        stepcount_str = f"step count (max): {logs['step_count'][-1]}"
        if i % 10 == 0:
            eval(env, actor, logs)
        pbar.set_description(", ".join([cum_reward_str, stepcount_str]))


def eval(env, actor, logs):
    with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
        eval_rollout = env.rollout(1000, actor)
        logs["eval reward"].append(eval_rollout["next", "reward"].mean().item())
        logs["eval reward (sum)"].append(eval_rollout["next", "reward"].sum().item())
        logs["eval step_count"].append(eval_rollout["step_count"].max().item())
        eval_str = (
            f"eval cumulative reward: {logs['eval reward (sum)'][-1]: 4.4f} "
            f"(init: {logs['eval reward (sum)'][0]: 4.4f}), "
            f"eval step-count: {logs['eval step_count'][-1]}"
        )
        logger.info(eval_str)
        del eval_rollout


def run_probing(loss_module, train_batch, test_batch):
    """Run probing."""
    probe_manager = SklearnProbeManager(LinearRegression, {}, lambda x, y: {"r2": r2_score(x, y)})

    with Probing(
        ".*",
        probe_manager.probe_factory,
        additional_keys=["labels", "step_type"],
        submodules_paths=["actor_network[0].module[0]", "critic_network"],
    ).prepare(loss_module) as hooked_module:
        train_batch["labels"] = train_batch["action"]
        train_batch["step_type"] = "fit"
        hooked_module(train_batch)

        test_batch["labels"] = test_batch["action"]
        test_batch["step_type"] = "predict"
        hooked_module(test_batch)

    # Add baseline: train on observations only
    baseline_probe = LinearRegression()
    train_obs = train_batch["observation"].detach().cpu().numpy()
    train_actions = train_batch["action"].detach().cpu().numpy()
    test_obs = test_batch["observation"].detach().cpu().numpy()
    test_actions = test_batch["action"].detach().cpu().numpy()

    baseline_probe.fit(train_obs, train_actions)
    baseline_predictions = baseline_probe.predict(test_obs)
    baseline_r2 = r2_score(test_actions, baseline_predictions)

    return {
        "fit_metrics": probe_manager.fit_metrics,
        "predict_metrics": probe_manager.predict_metrics,
        "baseline": {"r2": baseline_r2},
    }


def log_r2_scores(metrics: dict):
    """Log R² scores organized by network type."""

    def get_network_scores(network_type: str):
        """Helper function to get sorted scores for a network type."""
        scores = []
        for key, value in metrics.items():
            parsed_type, layer_num = parse_layer_number(key)
            if parsed_type == network_type and layer_num is not None:
                scores.append((layer_num, value["r2"]))
        scores.sort(key=lambda x: x[0])
        return scores

    network_configs = [
        ("baseline", "Baseline (Observations Only)"),
        ("actor", "Actor Network"),
        ("critic", "Critic Network"),
    ]

    for network_type, network_name in network_configs:
        scores = get_network_scores(network_type)
        if scores:
            if network_type == "baseline":
                logger.info(f"{network_name}: {scores[0][1]:.3f}")
            else:
                logger.info(f"{network_name} R² Scores:")
                for layer_num, r2_score in scores:
                    logger.info(f"  Layer {layer_num}: {r2_score:.3f}")


def parse_layer_number(key: str) -> tuple[str, int]:
    """Extract network type and layer number from probing key."""
    if key == "baseline":
        return "baseline", -1
    elif "actor_network" in key:
        match = re.search(r"actor_network\[0\]\.module\[0\]\.(\d+)_fwd", key)
        if match:
            return "actor", int(match.group(1))
    elif "critic_network" in key:
        match = re.search(r"critic_network\.(\d+)_fwd", key)
        if match:
            return "critic", int(match.group(1))
    return None, None


def plot_r2_scores(metrics: dict, baseline_score: float, save_path: str = None):
    """Plot R² scores across layers for both actor and critic networks."""

    def get_network_layers(network_type: str):
        """Helper function to get sorted layers for a network type."""
        layers = []
        for key, value in metrics.items():
            parsed_type, layer_num = parse_layer_number(key)
            if parsed_type == network_type and layer_num is not None:
                # Filter out layers 11 and 12 for critic network
                if network_type == "critic" and layer_num in [11, 12]:
                    continue
                layers.append((layer_num, value["r2"]))
        layers.sort(key=lambda x: x[0])
        return layers

    # Get layers for each network
    actor_layers = get_network_layers("actor")
    critic_layers = get_network_layers("critic")

    if not actor_layers and not critic_layers:
        logger.warning("No valid layer metrics found for plotting")
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    network_configs = [(actor_layers, "Actor Network", "o-", "C0"), (critic_layers, "Critic Network", "s-", "C1")]

    for layers, label, marker, color in network_configs:
        if layers:
            x = [layer[0] for layer in layers]
            y = [layer[1] for layer in layers]
            ax.plot(x, y, marker, label=label, linewidth=2, markersize=8, color=color)

    # Add observation baseline reference line
    ax.axhline(
        y=baseline_score,
        color="red",
        linestyle="--",
        alpha=0.7,
        label=f"Baseline (Observations): {baseline_score:.3f}",
    )

    ax.set_xlabel("Layer Number", fontsize=14)
    ax.set_ylabel("R² Score", fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.7)

    if actor_layers or critic_layers:
        all_y = [layer[1] for layer in actor_layers + critic_layers]
        all_y.append(baseline_score)
        y_min = min(all_y) if all_y else 0
        y_max = max(all_y) if all_y else 1
        y_range = y_max - y_min
        ax.set_ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved R2 plot to {save_path}")


def main(args):
    """Run reward probing demo."""
    logger.info("Starting reward probing demo...")

    env = get_env(args)
    actor, critic = get_policy_value(env)
    advantage_module, loss_module, optim = get_loss_advantage(actor, critic)
    replay_buffer = get_replay_buffer()
    collector = get_collector(env, actor)

    if args.train:
        train(env, actor, advantage_module, loss_module, optim, replay_buffer, collector)
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
        torch.save(actor.state_dict(), os.path.join(args.save_dir, "actor.pt"))
        torch.save(critic.state_dict(), os.path.join(args.save_dir, "critic.pt"))
    else:
        actor.load_state_dict(torch.load(os.path.join(args.save_dir, "actor.pt")))
        critic.load_state_dict(torch.load(os.path.join(args.save_dir, "critic.pt")))

    eval(env, actor, defaultdict(list))

    new_collector = iter(get_collector(env, actor))
    batch = next(new_collector)
    # batch = advantage_module(next(new_collector))
    indices = torch.randperm(batch.numel())
    split_idx = int(0.8 * batch.numel())
    train_indices, test_indices = indices[:split_idx], indices[split_idx:]
    train_batch = batch[train_indices]
    test_batch = batch[test_indices]

    advantage_module(train_batch)
    advantage_module(test_batch)

    metrics = run_probing(loss_module, train_batch, test_batch)

    log_r2_scores(metrics["fit_metrics"])
    log_r2_scores(metrics["predict_metrics"])

    plot_r2_scores(
        metrics["fit_metrics"], metrics["baseline"]["r2"], save_path=os.path.join(args.save_dir, "r2_scores_fit.png")
    )
    plot_r2_scores(
        metrics["predict_metrics"],
        metrics["baseline"]["r2"],
        save_path=os.path.join(args.save_dir, "r2_scores_predict.png"),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("value-saliency")
    parser.add_argument("--env_name", type=str, default="InvertedDoublePendulum-v4")
    parser.add_argument("--train", type=bool, default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--save_dir", type=str, default="results/torchrl/reward_probing")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
