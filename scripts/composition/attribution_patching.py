"""Attribution Patching

This script implements attribution patching by:
1. Running activation patching iteratively for different module types
2. Recording metrics for each position and layer
3. Visualizing the attribution results

Run with:
```
uv run --group scripts -m scripts.composition.attribution_patching
```
"""

import argparse
import re
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tensordict import TensorDict
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr
from loguru import logger
from tqdm import tqdm
from transformers.activations import NewGELUActivation
from transformers.models.gpt2.modeling_gpt2 import GPT2MLP, GPT2Block, GPT2Attention, GPT2Model
from torch.nn import LayerNorm, Embedding
from transformers.pytorch_utils import Conv1D

from tdhook.latent.activation_caching import ActivationCaching
from tdhook.latent.activation_patching import ActivationPatching
from tdhook.attribution.guided_backpropagation import GuidedBackpropagation
from tdhook.attribution.saliency import Saliency
from tdhook.attribution.integrated_gradients import IntegratedGradients
from tdhook.attribution.lrp import LRP
from tdhook.attribution.lrp_helpers.rules import Rule, BaseRuleMapper, LayerNormRule, PseudoIdentityRule, AHQKVRule
from tdhook.attribution.lrp_helpers.types import Linear

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class RelPMapper(BaseRuleMapper):
    def __init__(self):
        super().__init__()
        self._rules["ln"] = LayerNormRule()
        self._rules["pseudo_identity"] = PseudoIdentityRule()

    def _call(self, name: str, module: torch.nn.Module) -> Rule | None:
        if isinstance(module, (GPT2MLP, GPT2Block, GPT2Attention, GPT2Model, Embedding, Linear, Conv1D)):
            return self._rules["ignore"]
        elif isinstance(module, NewGELUActivation):
            return self._rules["pseudo_identity"]
        elif isinstance(module, LayerNorm):
            return self._rules["ln"]
        return super()._call(name, module)


class RelPAHMapper(RelPMapper):
    def __init__(self):
        super().__init__()
        self._rules["ah"] = AHQKVRule()

    def _call(self, name: str, module: torch.nn.Module) -> Rule | None:
        if isinstance(module, Conv1D) and name.endswith("c_attn"):
            return self._rules["ah"]
        return super()._call(name, module)


def to_numpy(tensor):
    if tensor.dtype == torch.bfloat16:
        tensor = tensor.float()
    return tensor.detach().cpu().numpy()


def split_kwargs(kwargs):
    custom_params = ["zmax", "zmin", "title", "xaxis", "yaxis", "facet_col", "facet_labels", "x", "save_to"]
    custom = {}
    plotly = {}

    for k, v in kwargs.items():
        if k in custom_params:
            custom[k] = v
        else:
            plotly[k] = v
    return custom, plotly


def embed_inputs(model, inputs):
    inputs_ids = inputs["input_ids"]
    inputs_embeds = model.transformer.wte(inputs_ids)
    return {"inputs_embeds": inputs_embeds, "attention_mask": inputs["attention_mask"]}


def imshow(array, **kwargs):
    array = to_numpy(array)
    custom_kwargs, plotly_kwargs = split_kwargs(kwargs)

    plt.switch_backend("Agg")
    plt.rcParams.update({"font.size": 12})

    if "facet_col" in custom_kwargs:
        n_facets = array.shape[custom_kwargs["facet_col"]]
        fig, axes = plt.subplots(1, n_facets, figsize=(5 * n_facets, 4))
        if n_facets == 1:
            axes = [axes]

        global_zmin = custom_kwargs.get("zmin", array.min())
        global_zmax = custom_kwargs.get("zmax", array.max())

        for i in range(n_facets):
            if custom_kwargs["facet_col"] == 0:
                data = array[i]
            else:
                data = array[:, i]

            im = axes[i].imshow(data, cmap="RdBu", aspect="auto", vmin=global_zmin, vmax=global_zmax)

            if "xaxis" in custom_kwargs:
                axes[i].set_xlabel(custom_kwargs["xaxis"])
            if "yaxis" in custom_kwargs and i == 0:
                axes[i].set_ylabel(custom_kwargs["yaxis"])

            axes[i].invert_yaxis()

            if "facet_labels" in custom_kwargs and i < len(custom_kwargs["facet_labels"]):
                axes[i].set_title(custom_kwargs["facet_labels"][i])

            if "x" in custom_kwargs:
                axes[i].set_xticks(range(len(custom_kwargs["x"])))
                axes[i].set_xticklabels(custom_kwargs["x"], rotation=60, ha="right")

        fig.subplots_adjust(right=0.85)
        fig.colorbar(im, ax=axes, shrink=0.8, pad=0.02)
    else:
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(array, cmap="RdBu_r", aspect="auto")

        if "zmax" in custom_kwargs or "zmin" in custom_kwargs:
            im.set_clim(custom_kwargs.get("zmin", array.min()), custom_kwargs.get("zmax", array.max()))

        if "xaxis" in custom_kwargs:
            ax.set_xlabel(custom_kwargs["xaxis"])
        if "yaxis" in custom_kwargs:
            ax.set_ylabel(custom_kwargs["yaxis"])

        if "x" in custom_kwargs:
            ax.set_xticks(range(len(custom_kwargs["x"])))
            ax.set_xticklabels(custom_kwargs["x"], rotation=60, ha="right")

        plt.colorbar(im, ax=ax, pad=0.02)

    if "title" in custom_kwargs:
        fig.suptitle(custom_kwargs["title"])

    if "save_to" in custom_kwargs:
        os.makedirs(os.path.dirname(custom_kwargs["save_to"]), exist_ok=True)
        plt.savefig(custom_kwargs["save_to"], dpi=150, bbox_inches="tight")
        print(f"Plot saved as '{custom_kwargs['save_to']}'")
        plt.close()
    else:
        plt.show()


def run_activation_patching(
    model, clean_inputs, corrupted_inputs, layer_map, metric_fn, progress_label="Activation patching"
):
    results = []
    inputs = TensorDict(**clean_inputs, batch_size=clean_inputs["input_ids"].shape[0])
    inputs["patched"] = TensorDict(**corrupted_inputs, batch_size=clean_inputs["input_ids"].shape[0])
    seq_len = clean_inputs["input_ids"].shape[1]

    n_layers = len(layer_map)
    total_iterations = n_layers * seq_len
    progress_bar = tqdm(total=total_iterations, desc=progress_label, unit="patch")

    for layer_num in range(n_layers):
        layer = (
            layer_map[str(layer_num)]
            .replace(f".{layer_num}.", f":{layer_num}:")
            .replace(f".{layer_num}", f":{layer_num}:")
        )
        pos = 0

        def pos_patch_fn(module_key, output, output_to_patch, **kwargs):
            nonlocal pos
            was_tuple = False
            if isinstance(output, tuple):
                was_tuple = True
                others = output[1:]
                output = output[0]

            patched_output = output.clone()
            patched_output[:, pos] = output_to_patch[:, pos]
            return patched_output if not was_tuple else (patched_output, *others)

        activation_patching = ActivationPatching(
            modules_to_patch=[layer], patch_fn=pos_patch_fn, cache_callback=callback
        )

        with activation_patching.prepare(
            model, in_keys={k: k for k in clean_inputs}, out_keys=["output"]
        ) as patched_model:
            position_metrics = []

            for pos in range(seq_len):
                with torch.no_grad():
                    output = patched_model(inputs)
                    metric = metric_fn(output["patched", "output", "logits"])
                    position_metrics.append(metric.item())

                progress_bar.update(1)

            results.append(torch.tensor(position_metrics))

    progress_bar.close()
    return torch.stack(results)


def get_logit_diff(logits, answer_token_indices):
    if len(logits.shape) >= 3:
        logits = logits[..., -1, :]
    _answer_token_indices = answer_token_indices.unsqueeze(-1)
    if len(logits.shape) == 3:
        _answer_token_indices = _answer_token_indices.unsqueeze(-1)
    correct_logits = logits.gather(-1, _answer_token_indices[:, 0])
    incorrect_logits = logits.gather(-1, _answer_token_indices[:, 1])
    return (correct_logits - incorrect_logits).mean()


def callback(**kwargs):
    if kwargs["direction"] == "fwd":
        if isinstance(kwargs.get("output"), tuple):
            return kwargs["output"][0]
        return kwargs["output"]
    elif kwargs["direction"] == "bwd":
        if isinstance(kwargs.get("grad_output"), tuple):
            return kwargs["grad_output"][0]
        return kwargs["grad_output"]
    raise ValueError(f"Invalid direction: {kwargs['direction']}")


def compute_correlations(activation_patching_results, attribution_results, method_name):
    """Compute Pearson correlations between activation patching and attribution results."""
    n_module_types = activation_patching_results.shape[0]
    module_names = ["Residual Stream", "Attention Output", "MLP Output"]
    correlations = []

    for module_idx in range(n_module_types):
        ap_data = activation_patching_results[module_idx].flatten()
        attr_data = attribution_results[module_idx].flatten()
        corr, _ = pearsonr(ap_data.numpy(), attr_data.detach().numpy())
        correlations.append(corr)

    return np.array(correlations), module_names


def plot_all_correlations(all_correlations, module_names, save_path):
    """Create grouped bar chart for all correlation results."""
    plt.switch_backend("Agg")
    plt.rcParams.update({"font.size": 16})
    method_colors = {
        "relp": "#1f77b4",
        "relp_ah": "#ff7f0e",
        "saliency": "#2ca02c",
        "guided_backpropagation": "#9467bd",
        "integrated_gradients": "#8c564b",
    }
    method_labels = {
        "relp": "RelP",
        "relp_ah": "RelP+AH",
        "saliency": "Saliency",
        "guided_backpropagation": "Guided Backpropagation",
        "integrated_gradients": "Integrated Gradients",
    }

    n_methods = len(all_correlations)
    n_modules = len(module_names)
    fig, ax = plt.subplots(figsize=(14, 8))
    bar_width = 0.12
    group_width = bar_width * (n_methods + 1)
    x_pos = np.arange(n_modules) * group_width
    for i, (method_name, correlations) in enumerate(all_correlations.items()):
        offset = (i - n_methods / 2 + 0.5) * bar_width
        ax.bar(
            x_pos + offset,
            correlations,
            bar_width,
            label=method_labels[method_name],
            color=method_colors.get(method_name, f"C{i}"),
            alpha=0.8,
        )

        for j, corr in enumerate(correlations):
            ax.text(x_pos[j] + offset, corr + 0.02, f"{corr:.3f}", ha="center", va="bottom", fontsize=10, rotation=0)

    ax.set_xlabel("Module Type")
    ax.set_ylabel("Pearson Correlation Coefficient")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(module_names, ha="center")
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    logger.info(f"All correlations plot saved as '{save_path}'")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Attribution Patching with tdhook")
    parser.add_argument("--model_name", default="gpt2", help="Model name to use")
    parser.add_argument("--output_dir", default="results/attribution_patching", help="Output directory")
    parser.add_argument("--device", default="cpu", help="Device to use")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    logger.info("Loading model and tokenizer...")
    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model.to(args.device)
    model.eval()

    prompts = [
        "When John and Mary went to the shops, John gave the bag to",
        "When John and Mary went to the shops, Mary gave the bag to",
        "When Tom and James went to the park, James gave the ball to",
        "When Tom and James went to the park, Tom gave the ball to",
        "When Dan and Sid went to the shops, Sid gave an apple to",
        "When Dan and Sid went to the shops, Dan gave an apple to",
        "After Martin and Amy went to the park, Amy gave a drink to",
        "After Martin and Amy went to the park, Martin gave a drink to",
    ]
    answers = [
        (" Mary", " John"),
        (" John", " Mary"),
        (" Tom", " James"),
        (" James", " Tom"),
        (" Dan", " Sid"),
        (" Sid", " Dan"),
        (" Martin", " Amy"),
        (" Amy", " Martin"),
    ]

    clean_inputs = tokenizer(prompts, return_tensors="pt")
    clean_tokens = clean_inputs["input_ids"]
    corrupted_tokens = clean_tokens[[(i + 1 if i % 2 == 0 else i - 1) for i in range(len(clean_tokens))]]
    corrupted_inputs = {"input_ids": corrupted_tokens, "attention_mask": clean_inputs["attention_mask"]}

    answer_token_indices = torch.tensor(
        [[tokenizer.encode(answers[i][j])[0] for j in range(2)] for i in range(len(answers))],
    )

    logger.info("Setting up activation caching...")
    context = ActivationCaching(
        key_pattern="^(?!transformer$).*", callback=callback, relative=True, directions=["fwd", "bwd"]
    )
    caching_context = context.prepare(model, in_keys={k: k for k in clean_inputs}, out_keys=["output"])

    with caching_context as hooked_model:
        shuttle = TensorDict(**clean_inputs, batch_size=clean_inputs["input_ids"].shape[0])
        shuttle = hooked_model(shuttle)
        clean_cache = caching_context.cache.clone()
        clean_logits = shuttle["output", "logits"]

        shuttle = TensorDict(**corrupted_inputs, batch_size=clean_inputs["input_ids"].shape[0])
        shuttle = hooked_model(shuttle)
        corrupted_cache = caching_context.cache.clone()
        corrupted_logits = shuttle["output", "logits"]

    clean_logit_diff = get_logit_diff(clean_logits, answer_token_indices).item()
    corrupted_logit_diff = get_logit_diff(corrupted_logits, answer_token_indices).item()

    logger.info(f"Clean logit diff: {clean_logit_diff:.4f}")
    logger.info(f"Corrupted logit diff: {corrupted_logit_diff:.4f}")

    CLEAN_BASELINE = clean_logit_diff
    CORRUPTED_BASELINE = corrupted_logit_diff

    def ioi_metric(logits, answer_token_indices=answer_token_indices):
        return (get_logit_diff(logits, answer_token_indices) - CORRUPTED_BASELINE) / (
            CLEAN_BASELINE - CORRUPTED_BASELINE
        )

    mlp_map = {
        re.match(r"transformer\.h\.(\d+)\.mlp", k).group(1): k
        for k in caching_context.cache["fwd"].keys()
        if re.match(r"transformer\.h\.\d+\.mlp", k)
    }
    attn_map = {
        re.match(r"transformer\.h\.(\d+)\.attn", k).group(1): k
        for k in caching_context.cache["fwd"].keys()
        if re.match(r"transformer\.h\.\d+\.attn", k)
    }
    resid_map = {
        re.match(r"transformer\.h\.(\d+)", k).group(1): k
        for k in caching_context.cache["fwd"].keys()
        if re.match(r"transformer\.h\.\d+", k)
    }

    all_results = {}
    for attribution_method in ["relp", "relp_ah", "saliency", "guided_backpropagation", "integrated_gradients"]:
        if attribution_method == "saliency":
            attribution_class = Saliency
            method_kwargs = {}
        elif attribution_method == "guided_backpropagation":
            attribution_class = GuidedBackpropagation
            method_kwargs = {}
        elif attribution_method == "integrated_gradients":
            attribution_class = IntegratedGradients

            def init_attr_inputs(inputs, _):
                return inputs.select("inputs_embeds")

            method_kwargs = {"init_attr_inputs": init_attr_inputs, "n_steps": 10}
        elif attribution_method == "relp":
            attribution_class = LRP
            method_kwargs = {"rule_mapper": RelPMapper(), "skip_modules": LRP.default_skip}
        elif attribution_method == "relp_ah":
            attribution_class = LRP
            method_kwargs = {"rule_mapper": RelPAHMapper(), "skip_modules": LRP.default_skip}
        else:
            raise ValueError(f"Unknown attribution method: {attribution_method}")

        def init_attr_targets(targets, _):
            logits = targets["output", "logits"]
            return TensorDict(metric=ioi_metric(logits))

        attribution_kwargs = {
            "init_attr_targets": init_attr_targets,
            "clean_intermediate_keys": False,
            "use_inputs": False,
            "input_modules": [
                *(mlp_map[k].replace(f".{k}.", f":{k}:") for k in mlp_map),
                *(attn_map[k].replace(f".{k}.", f":{k}:") for k in attn_map),
                *(resid_map[k].replace(f".{k}", f":{k}:") for k in resid_map),
            ],
            "cache_callback": callback,
            **method_kwargs,
        }
        attribution_factory = attribution_class(**attribution_kwargs)

        logger.info(f"Computing attribution results for {attribution_method}...")
        _clean_inputs = (
            embed_inputs(model, clean_inputs) if attribution_method == "integrated_gradients" else clean_inputs
        )
        _corrupted_inputs = (
            embed_inputs(model, corrupted_inputs) if attribution_method == "integrated_gradients" else corrupted_inputs
        )
        with attribution_factory.prepare(
            model, in_keys={k: k for k in _clean_inputs}, out_keys=["output"]
        ) as hooked_model:
            shuttle = TensorDict(**_corrupted_inputs, batch_size=_clean_inputs["attention_mask"].shape[0])
            if attribution_method == "integrated_gradients":
                shuttle[("baseline", "inputs_embeds")] = _clean_inputs["inputs_embeds"]
            shuttle = hooked_model(shuttle)
            corrupted_grads = shuttle["attr"]
            for k in list(corrupted_grads.keys()):
                corrupted_grads.rename_key_(k, re.sub(r":(\d+):", r".\1.", k).rstrip("."))
            select_keys = corrupted_grads.keys()

        attribution_cache = corrupted_grads * (
            clean_cache["fwd"].select(*select_keys) - corrupted_cache["fwd"].select(*select_keys)
        )
        n_layers = len(mlp_map)

        resid_attribution = torch.stack([attribution_cache.get(resid_map[str(i)]) for i in range(n_layers)]).sum(
            dim=(1, 3)
        )
        attn_attribution = torch.stack([attribution_cache.get(attn_map[str(i)]) for i in range(n_layers)]).sum(
            dim=(1, 3)
        )
        mlp_attribution = torch.stack([attribution_cache.get(mlp_map[str(i)]) for i in range(n_layers)]).sum(
            dim=(1, 3)
        )

        attribution_results = torch.stack([resid_attribution, attn_attribution, mlp_attribution], dim=0)

        logger.info(f"Generating gradient-based attribution plot for {attribution_method}...")
        imshow(
            attribution_results,
            facet_col=0,
            facet_labels=["MLP Output", "Attn Output", "Residual Stream"],
            xaxis="Position",
            yaxis="Layer",
            zmax=1,
            zmin=-1,
            x=[f"{tokenizer.decode(tok)}_{i}" for i, tok in enumerate(clean_tokens[0])],
            save_to=os.path.join(args.output_dir, f"{attribution_method}_patching_plot.png"),
        )

        all_results[attribution_method] = attribution_results

    logger.info("Running activation patching...")
    resid_results = run_activation_patching(
        model, clean_inputs, corrupted_inputs, resid_map, ioi_metric, "Residual Stream"
    )
    attn_results = run_activation_patching(
        model, clean_inputs, corrupted_inputs, attn_map, ioi_metric, "Attention Output"
    )
    mlp_results = run_activation_patching(model, clean_inputs, corrupted_inputs, mlp_map, ioi_metric, "MLP Output")

    activation_patching_results = torch.stack([resid_results, attn_results, mlp_results], dim=0)

    logger.info("Generating activation patching plot...")
    imshow(
        activation_patching_results,
        facet_col=0,
        facet_labels=["Residual Stream", "Attention Output", "MLP Output"],
        xaxis="Position",
        yaxis="Layer",
        zmax=1,
        zmin=-1,
        x=[f"{tokenizer.decode(tok)}_{i}" for i, tok in enumerate(clean_tokens[0])],
        save_to=os.path.join(args.output_dir, "activation_patching_plot.png"),
    )

    logger.info("Computing correlations between activation patching and attribution methods...")
    all_correlations = {}
    module_names = None
    for method_name, attribution_results in all_results.items():
        correlations, module_names = compute_correlations(
            activation_patching_results, attribution_results, method_name
        )
        all_correlations[method_name] = correlations

        logger.info(f"{method_name.upper()} correlations:")
        for i, corr in enumerate(correlations):
            logger.info(f"  {module_names[i]}: {corr:.3f}")
    all_correlations_path = os.path.join(args.output_dir, "all_correlations_plot.png")
    plot_all_correlations(all_correlations, module_names, all_correlations_path)


if __name__ == "__main__":
    main()
