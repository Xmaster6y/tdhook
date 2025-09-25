"""Concept Attribution using tdhook.

This script implements concept attribution by:
1. Collecting activations from a specified layer
2. Training a single linear probe (CAV) on those activations for a specific concept
3. Computing concept attribution (sensitivity × CAV) on a single example
4. Running randomization test to validate the concept attribution

Run with:
```
uv run --group scripts -m scripts.composition.concept_attribution
```
"""

import argparse
import os

import torch
import timm
from datasets import load_dataset
import numpy as np
from sklearn.metrics import accuracy_score
from loguru import logger
import matplotlib.pyplot as plt
from tensordict import TensorDict
from torch.utils.data import DataLoader

from tdhook.latent.activation_caching import ActivationCaching
from tdhook.attribution.saliency import Saliency
from tdhook.attribution.integrated_gradients import IntegratedGradients
from tdhook.attribution.grad_cam import GradCAM, DimsConfig
from tdhook.attribution.lrp import LRP
from tdhook.attribution.lrp_helpers.rules import EpsilonPlus, Rule

from .concept_utils import load_and_preprocess_image, create_balanced_dataset, collect_activations, train_probe


class CustomEpsilonPlus(EpsilonPlus):
    def _call(self, name: str, module: torch.nn.Module) -> Rule | None:
        if isinstance(module, (timm.models.vgg.ConvMlp, timm.layers.ClassifierHead, timm.layers.SelectAdaptivePool2d)):
            return self._rules["ignore"]
        return super()._call(name, module)


def get_sensitivity_factory(sensitivity_method, layer_number, target_class, grad_callback=None, record_layer=False):
    def init_attr_targets(targets, _):
        output = targets["output"]
        best_logit = output[..., target_class]
        return TensorDict(out=best_logit, batch_size=targets.batch_size)

    if sensitivity_method == "saliency":
        sensitivity_class = Saliency
        method_kwargs = {"absolute": True}
    elif sensitivity_method == "inputxgradient":
        sensitivity_class = Saliency
        method_kwargs = {"absolute": True, "multiply_by_inputs": True}
    elif sensitivity_method == "integrated_gradients":
        sensitivity_class = IntegratedGradients
        method_kwargs = {}
    elif sensitivity_method == "grad_cam":
        sensitivity_class = GradCAM
        method_kwargs = {
            "modules_to_attribute": {
                f"features:{layer_number}:": DimsConfig(weight_pooling_dims=(2, 3), feature_sum_dims=(1,))
            }
        }
    elif sensitivity_method == "lrp":
        sensitivity_class = LRP
        method_kwargs = {"rule_mapper": CustomEpsilonPlus(epsilon=1e-6), "skip_modules": LRP.default_skip}
    else:
        raise ValueError(f"Unknown sensitivity method: {sensitivity_method}")

    sensitivity_kwargs = {"init_attr_targets": init_attr_targets, "clean_intermediate_keys": False, **method_kwargs}
    if grad_callback is not None:
        sensitivity_kwargs["output_grad_callbacks"] = {f"features:{layer_number}:": grad_callback}
    if record_layer:
        sensitivity_kwargs["input_modules"] = [f"features:{layer_number}:"]
    return sensitivity_class(
        **sensitivity_kwargs,
    )


def get_sensitivity(
    input_td, hooked_model, sensitivity_method, layer_number, device, use_abs=False, record_layer=False
):
    if sensitivity_method == "integrated_gradients":
        input_td[("baseline", "input")] = torch.zeros_like(input_td["input"]).unsqueeze(0).to(device)
    output = hooked_model(input_td)
    if sensitivity_method == "grad_cam":
        sensitivity = output.get(("attr", f"features:{layer_number}:")).squeeze(0)
        sensitivity = upscale_heatmap_to_image_size(sensitivity, *input_td["input"].shape[1:])
    elif record_layer:
        sensitivity = output.get(("attr", f"features:{layer_number}:")).sum(dim=(2, 3))
    else:
        sensitivity = output.get(("attr", "input")).squeeze(0).sum(dim=0)

    if use_abs:
        sensitivity = sensitivity.abs()
    return sensitivity


def collect_relevances(
    model, layer_number, target_class, concept_dataset, sensitivity_method, device="cpu", batch_size=32
):
    logger.info(f"Collecting activations from features.{layer_number} with batch size {batch_size}...")

    dataloader = DataLoader(
        concept_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    relevances_list = []
    labels_list = []
    sensitivity_factory = get_sensitivity_factory(sensitivity_method, layer_number, target_class, record_layer=True)

    with sensitivity_factory.prepare(model) as hooked_model:
        for batch_idx, batch in enumerate(dataloader):
            batch_tensor = batch["image"].to(device)

            input_td = TensorDict({"input": batch_tensor}, batch_size=batch_tensor.shape[0])
            relevances = get_sensitivity(
                input_td, hooked_model, sensitivity_method, layer_number, device, record_layer=True, use_abs=True
            )

            relevances_list.append(relevances.cpu())
            labels_list.append(batch["label"].cpu())

            logger.info(f"Processed batch {batch_idx + 1}/{len(dataloader)}")

    data = TensorDict(
        {
            "relevances": torch.cat(relevances_list, dim=0),
            "labels": torch.cat(labels_list, dim=0),
        }
    )
    return data


def get_grad_callback(select_method, data=None, probe=None, device="cpu"):
    if select_method.startswith("actmax"):
        n_channels = int(select_method.split("_")[1])
        importance = get_channel_relative_importance(data, "activations")
        channels = importance.argsort()[-n_channels:]
    elif select_method.startswith("relmax"):
        n_channels = int(select_method.split("_")[1])
        importance = get_channel_relative_importance(data, "relevances")
        channels = importance.argsort()[-n_channels:]
    else:
        channels = None

    if channels is not None:

        def grad_callback(grad_output, **kwargs):
            _grad_output = grad_output[0]
            nonlocal channels, device
            channel_mask = torch.zeros((1, _grad_output.shape[1], 1, 1), device=device)
            channel_mask[0, channels] = 1
            return (_grad_output * channel_mask,)

    else:

        def grad_callback(grad_output, **kwargs):
            _grad_output = grad_output[0]
            nonlocal probe
            # dot_product = (_grad_output * probe._coef[..., None, None]).sum(1, keepdims=True)
            # new_grad = dot_product * (dot_product > 0) * probe._coef[..., None, None]

            dot_product = (_grad_output * probe._coef[..., None, None]).sum(1, keepdims=True)
            new_grad = _grad_output * (dot_product > 0)

            # new_grad = _grad_output * probe._coef[..., None, None]

            new_grad = new_grad.to(_grad_output.dtype)
            return (new_grad,)

    return grad_callback


def evaluate_probe(activations_cache, probe):
    """Evaluate the trained probe on the test set."""
    logger.info("Evaluating probe on test set...")

    activations = activations_cache["activations"]
    labels = activations_cache["labels"]

    batch_size = activations.shape[0]
    flattened_activations = activations.view(batch_size, -1).numpy()
    concept_labels = np.array(labels, dtype=int)

    predictions = probe.predict(flattened_activations)
    accuracy = accuracy_score(concept_labels, predictions)

    from sklearn.metrics import precision_score, recall_score, f1_score

    precision = precision_score(concept_labels, predictions, zero_division=0)
    recall = recall_score(concept_labels, predictions, zero_division=0)
    f1 = f1_score(concept_labels, predictions, zero_division=0)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "total_samples": len(concept_labels),
        "positive_samples": int(np.sum(concept_labels)),
        "negative_samples": int(len(concept_labels) - np.sum(concept_labels)),
    }


def get_channel_relative_importance(cache, importance_key):
    activations = cache[importance_key]
    positive_activations = activations[cache["labels"] == 1]
    negative_activations = activations[cache["labels"] == 0]
    positive_importance = positive_activations.mean(dim=0)
    negative_importance = negative_activations.mean(dim=0)
    return positive_importance - negative_importance


def upscale_heatmap_to_image_size(heatmap, target_height, target_width):
    heatmap_4d = heatmap.unsqueeze(0).unsqueeze(0)
    upscaled_4d = torch.nn.functional.interpolate(
        heatmap_4d, size=(target_height, target_width), mode="bilinear", align_corners=False
    )
    return upscaled_4d.squeeze(0).squeeze(0)


def compute_concept_attribution(
    model, layer_number, image, target_class, grad_callback, sensitivity_method="saliency", device="cuda"
):
    sensitivity = get_sensitivity_factory(sensitivity_method, layer_number, target_class)
    with sensitivity.prepare(model) as hooked_model:
        input_td = TensorDict({"input": image.unsqueeze(0).to(device)}, batch_size=1)
        input_sensitivity = get_sensitivity(
            input_td, hooked_model, sensitivity_method, layer_number, device, use_abs=True
        )

    sensitivity = get_sensitivity_factory(sensitivity_method, layer_number, target_class, grad_callback)
    with sensitivity.prepare(model) as hooked_model:
        input_td = TensorDict({"input": image.unsqueeze(0).to(device)}, batch_size=1)
        input_cond_sensitivity = get_sensitivity(
            input_td, hooked_model, sensitivity_method, layer_number, device, use_abs=True
        )

    return input_sensitivity, input_cond_sensitivity


def visualize_concept_attribution(
    image, concept, input_sensitivity, input_cond_sensitivity, sensitivity_method, save_path=None
):
    def normalise_heatmap(data):
        abs_max = np.abs(data).max()
        if abs_max > 0:
            return data / abs_max
        return data

    plt.rcParams.update({"font.size": 14})
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    image_np = image.permute(1, 2, 0).cpu().numpy()
    image_np = (image_np - image_np.min()) / (image_np.max() - image_np.min())
    axes[0].imshow(image_np)
    axes[0].axis("off")

    sensitivity_vis = input_sensitivity.cpu().numpy()
    sensitivity_normalized = normalise_heatmap(sensitivity_vis)
    axes[1].imshow(sensitivity_normalized, cmap="bwr", vmin=-1, vmax=1)
    axes[1].axis("off")

    cav_vis = input_cond_sensitivity.cpu().detach().numpy()
    cav_normalized = normalise_heatmap(cav_vis)
    axes[2].imshow(cav_normalized, cmap="bwr", vmin=-1, vmax=1)
    axes[2].axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Visualization saved to {save_path}")

    plt.close()


def flip_pixels_by_attribution(image, attribution_heatmap, flip_percentage, flip_method="mean"):
    if flip_percentage == 0:
        return image.clone()
    elif flip_percentage == 100:
        if flip_method == "white":
            flip_value = 1.0
        elif flip_method == "black":
            flip_value = 0.0
        elif flip_method == "mean":
            flip_value = image.mean()
        else:
            raise ValueError(f"Unknown flip method: {flip_method}")

        baseline_image = torch.full_like(image, flip_value)
        return baseline_image

    if attribution_heatmap.dim() == 3:
        attribution_heatmap = attribution_heatmap.sum(dim=0)

    image_height, image_width = image.shape[1], image.shape[2]
    heatmap_height, heatmap_width = attribution_heatmap.shape

    if heatmap_height != image_height or heatmap_width != image_width:
        logger.info(f"Upscaling heatmap from {heatmap_height}x{heatmap_width} to {image_height}x{image_width}")
        attribution_heatmap = upscale_heatmap_to_image_size(attribution_heatmap, image_height, image_width)

    total_pixels = attribution_heatmap.numel()
    num_pixels_to_flip = int(total_pixels * flip_percentage / 100)

    flat_heatmap = attribution_heatmap.flatten()
    _, top_indices = torch.topk(flat_heatmap, num_pixels_to_flip)

    height, width = attribution_heatmap.shape
    row_indices = top_indices // width
    col_indices = top_indices % width

    modified_image = image.clone()

    if flip_method == "white":
        flip_value = 1.0
    elif flip_method == "black":
        flip_value = 0.0
    elif flip_method == "mean":
        flip_value = image.mean()
    else:
        raise ValueError(f"Unknown flip method: {flip_method}")

    for i in range(len(row_indices)):
        modified_image[:, row_indices[i], col_indices[i]] = flip_value

    return modified_image


def flip_pixels_by_attribution_batch(image, attribution_heatmap, flip_percentages, flip_method="mean"):
    modified_images = []

    for flip_pct in flip_percentages:
        modified_image = flip_pixels_by_attribution(image, attribution_heatmap, flip_pct, flip_method)
        modified_images.append(modified_image)

    return torch.stack(modified_images)


def evaluate_attribution_robustness(
    model,
    image,
    attribution_heatmap,
    probe,
    layer_number,
    target_class,
    flip_percentages,
    flip_method="mean",
    device="cuda",
):
    with torch.no_grad():
        original_output = model(image.unsqueeze(0).to(device))
        original_logit = original_output[0, target_class].item()

        with ActivationCaching(f"features.{layer_number}").prepare(model) as hooked_model:
            input_td = TensorDict({"input": image.unsqueeze(0).to(device)}, batch_size=1)
            hooked_model(input_td)
            cache = hooked_model.hooking_context.cache
            original_concept_activations = cache[f"features.{layer_number}"].amax(dim=(2, 3))

        original_concept_pred = probe.predict(original_concept_activations.view(1, -1).cpu().numpy())[0]
        original_concept_proba = probe.predict_proba(original_concept_activations.view(1, -1).cpu().numpy())[0]

        logger.info(f"Generating {len(flip_percentages)} modified images...")
        modified_images_batch = flip_pixels_by_attribution_batch(
            image, attribution_heatmap, flip_percentages, flip_method
        )

        logger.info("Running model on batch of modified images...")
        modified_outputs = model(modified_images_batch.to(device))
        modified_logits = modified_outputs[:, target_class]

        with ActivationCaching(f"features.{layer_number}").prepare(model) as hooked_model:
            input_td = TensorDict({"input": modified_images_batch.to(device)}, batch_size=len(flip_percentages))
            hooked_model(input_td)
            cache = hooked_model.hooking_context.cache
            modified_concept_activations_batch = cache[f"features.{layer_number}"].amax(dim=(2, 3))

        modified_concept_preds = probe.predict(
            modified_concept_activations_batch.view(len(flip_percentages), -1).cpu().numpy()
        )
        modified_concept_probas = probe.predict_proba(
            modified_concept_activations_batch.view(len(flip_percentages), -1).cpu().numpy()
        )

    results = {
        "original_logit": original_logit,
        "original_concept_pred": original_concept_pred,
        "original_concept_proba": original_concept_proba,
        "flip_results": {},
    }

    for i, flip_pct in enumerate(flip_percentages):
        modified_logit = modified_logits[i].item()
        modified_concept_pred = modified_concept_preds[i]
        modified_concept_proba = modified_concept_probas[i]

        logit_drop = original_logit - modified_logit
        concept_drop = original_concept_proba[1] - modified_concept_proba[1]

        relative_logit_drop = logit_drop / abs(original_logit) if original_logit != 0 else 0
        relative_concept_drop = concept_drop / abs(original_concept_proba[1]) if original_concept_proba[1] != 0 else 0

        results["flip_results"][flip_pct] = {
            "modified_logit": modified_logit,
            "logit_drop": logit_drop,
            "modified_concept_pred": modified_concept_pred,
            "modified_concept_proba": modified_concept_proba,
            "concept_drop": concept_drop,
            "relative_logit_drop": relative_logit_drop,
            "relative_concept_drop": relative_concept_drop,
        }

    return results


def batch_evaluate_attribution_robustness(
    model,
    image,
    attribution_heatmap,
    probe,
    layer_number,
    target_class,
    flip_percentages,
    flip_method="mean",
    device="cuda",
):
    logger.info("Evaluating attribution robustness for single image...")

    result = evaluate_attribution_robustness(
        model, image, attribution_heatmap, probe, layer_number, target_class, flip_percentages, flip_method, device
    )

    return result


def format_method_label(method):
    replacements = {
        "lrp": "LRP",
        "inputxgradient": "In*Grad",
        "cav": "CAV",
        "actmax": "ActMax",
        "saliency": "Saliency",
        "integrated_gradients": "Int. Grad",
        "grad_cam": "Grad CAM",
    }

    select_method, sensitivity_method = method.split(":")

    for pattern, replacement in replacements.items():
        if sensitivity_method.lower() == pattern:
            sensitivity_method = replacement
        if select_method.lower().startswith(pattern):
            select_method = select_method.replace(pattern, replacement, 1)

    return f"{sensitivity_method} + {select_method}"


def plot_evaluation_results(evaluation_results, save_path=None):
    logger.info("Plotting evaluation results...")
    plt.rcParams.update({"font.size": 12})

    first_image = list(evaluation_results.keys())[0]
    first_method = list(evaluation_results[first_image].keys())[0]
    flip_percentages = list(evaluation_results[first_image][first_method]["flip_results"].keys())

    image_names = list(evaluation_results.keys())
    method_combinations = list(evaluation_results[first_image].keys())
    method_combinations.sort(key=lambda x: x.split(":")[::-1])

    # Create consistent color and marker mapping for method combinations
    colors = ["b", "r", "g", "m", "c", "y", "k", "orange", "purple", "brown", "pink", "gray", "olive"]
    markers = ["o", "s", "^", "v", "D", "p", "*", "h", "H", "+", "x", "|"]

    # Create mapping for consistent colors and markers
    method_to_style = {}
    sensitivity_methods = list(set([method.split(":")[1] for method in method_combinations]))
    for i, method in enumerate(method_combinations):
        marker_idx = sensitivity_methods.index(method.split(":")[1])
        method_to_style[method] = {"color": colors[i % len(colors)], "marker": markers[marker_idx % len(markers)]}

    # Calculate averaged results across all images
    averaged_results = {}
    for method in method_combinations:
        averaged_results[method] = {
            "relative_logit_drops": [],
            "relative_concept_drops": [],
            "relative_logit_drops_std": [],
            "relative_concept_drops_std": [],
        }

        for flip_pct in flip_percentages:
            logit_drops = []
            concept_drops = []

            for image_name in image_names:
                result = evaluation_results[image_name][method]
                logit_drops.append(result["flip_results"][flip_pct]["relative_logit_drop"])
                concept_drops.append(result["flip_results"][flip_pct]["relative_concept_drop"])

            # Average and std across images
            averaged_results[method]["relative_logit_drops"].append(np.mean(logit_drops))
            averaged_results[method]["relative_concept_drops"].append(np.mean(concept_drops))
            averaged_results[method]["relative_logit_drops_std"].append(np.std(logit_drops))
            averaged_results[method]["relative_concept_drops_std"].append(np.std(concept_drops))

    # Create single plot with averaged results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Plot averaged relative logit drops
    for method in method_combinations:
        style = method_to_style[method]
        means = averaged_results[method]["relative_logit_drops"]
        stds = averaged_results[method]["relative_logit_drops_std"]

        ax1.plot(
            flip_percentages,
            means,
            color=style["color"],
            marker=style["marker"],
            linestyle="-",
            linewidth=2,
            markersize=6,
            label=format_method_label(method),
        )

        ax1.fill_between(
            flip_percentages,
            np.array(means) - np.array(stds),
            np.array(means) + np.array(stds),
            color=style["color"],
            alpha=0.2,
        )

    ax1.set_xlabel("Pixel Flip Percentage (%)")
    ax1.set_ylabel("Relative Logit Drop (Averaged)")
    ax1.grid(True, alpha=0.3)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    # Plot averaged relative concept drops
    for method in method_combinations:
        style = method_to_style[method]
        means = averaged_results[method]["relative_concept_drops"]
        stds = averaged_results[method]["relative_concept_drops_std"]

        ax2.plot(
            flip_percentages,
            means,
            color=style["color"],
            marker=style["marker"],
            linestyle="-",
            linewidth=2,
            markersize=6,
            label=format_method_label(method),
        )

        ax2.fill_between(
            flip_percentages,
            np.array(means) - np.array(stds),
            np.array(means) + np.array(stds),
            color=style["color"],
            alpha=0.2,
        )

    ax2.set_xlabel("Pixel Flip Percentage (%)")
    ax2.set_ylabel("Relative Concept Drop (Averaged)")
    ax2.grid(True, alpha=0.3)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Evaluation plot saved to {save_path}")

    return fig


def main():
    parser = argparse.ArgumentParser(description="Concept Attribution with tdhook")
    parser.add_argument("--layer_number", default=28, help="Layer number to analyze")
    parser.add_argument("--concept", default="striped", help="Concept to analyze")
    parser.add_argument("--select_method", default="cav", choices=["actmax", "cav", "relmax"], help="Select method")
    parser.add_argument(
        "--cav_method",
        default="mean_difference",
        choices=["mean_difference", "logistic_regression_l1", "logistic_regression_l2"],
        help="CAV method",
    )
    parser.add_argument("--device", default="cpu", help="Device to use")
    parser.add_argument(
        "--num_samples_per_class", type=int, default=50, help="Number of samples per class (positive/negative)"
    )
    parser.add_argument("--batch_size", type=int, default=50, help="Batch size for activation collection")
    parser.add_argument("--output_dir", default="results/concept_attribution", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    logger.info("Loading VGG16 model...")
    model = timm.create_model("vgg16.tv_in1k", pretrained=True)
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.ReLU):
            *parts, last = name.split(".")
            submodule = model
            for part in parts:
                submodule = getattr(submodule, part)
            setattr(submodule, last, torch.nn.ReLU(inplace=False))
    model.to(args.device)
    model.eval()

    logger.info("Loading DTD dataset...")
    train_dataset = load_dataset("tanganke/dtd", split="train")
    test_dataset = load_dataset("tanganke/dtd", split="test")

    concept_names = train_dataset.features["label"].names
    concept_index = concept_names.index(args.concept)

    data_config = timm.data.resolve_model_data_config(model)
    transforms = timm.data.create_transform(**data_config, is_training=False)

    logger.info(f"Creating balanced training dataset for concept '{args.concept}'...")
    train_concept_dataset = create_balanced_dataset(
        train_dataset,
        concept_index,
        args.concept,
        transforms,
        num_samples_per_class=args.num_samples_per_class,
        seed=args.seed,
    )

    train_activations_cache = collect_activations(
        model, args.layer_number, train_concept_dataset, args.device, args.batch_size
    )
    probe = train_probe(train_activations_cache, args.concept, args.cav_method)

    test_concept_dataset = create_balanced_dataset(
        test_dataset,
        concept_index,
        args.concept,
        transforms,
        num_samples_per_class=args.num_samples_per_class,
        seed=args.seed,
    )
    test_activations_cache = collect_activations(
        model, args.layer_number, test_concept_dataset, args.device, args.batch_size
    )
    test_results = evaluate_probe(
        test_activations_cache,
        probe,
    )

    logger.info("=" * 50)
    logger.info("PROBE PERFORMANCE METRICS")
    logger.info("=" * 50)
    logger.info(f"Accuracy: {test_results['accuracy']:.3f}")
    logger.info(f"Precision: {test_results['precision']:.3f}")
    logger.info(f"Recall: {test_results['recall']:.3f}")
    logger.info(f"F1-score: {test_results['f1']:.3f}")
    logger.info(
        f"Test samples: {test_results['total_samples']} ({test_results['positive_samples']} positive, {test_results['negative_samples']} negative)"
    )
    logger.info("=" * 50)

    train_channel_relative_importance = get_channel_relative_importance(train_activations_cache, "activations")
    test_channel_relative_importance = get_channel_relative_importance(test_activations_cache, "activations")
    train_best_channels = train_channel_relative_importance.argsort()[-2:]
    test_best_channels = test_channel_relative_importance.argsort()[-2:]
    logger.info(f"Train best channels: {train_best_channels}")
    logger.info(f"Test best channels: {test_best_channels}")

    cav = probe.coef_.reshape(train_activations_cache["activations"].shape[1:])
    logger.info(f"Pooled CAV best channels: {cav.argsort()[-2:]}")

    evaluation_results = {}
    for image_name, target_class in [
        ("lemur_1", 383),
        ("zebra_1", 340),
        ("skunk_1", 361),
        ("lemur_2", 383),
        ("zebra_2", 340),
        ("skunk_2", 361),
    ]:
        logger.info(f"Using custom test image for visualization: {image_name}")
        test_image_path = os.path.join("results/concept_attribution", image_name + ".jpg")
        test_image = load_and_preprocess_image(test_image_path, transforms, args.device)
        evaluation_results[image_name] = {}
        for select_method in ["relmax_1", "actmax_1", "actmax_2", "cav"]:
            for sensitivity_method in ["saliency", "inputxgradient", "lrp"]:
                if select_method.startswith("relmax"):
                    train_data = collect_relevances(
                        model,
                        args.layer_number,
                        target_class,
                        train_concept_dataset,
                        sensitivity_method,
                        args.device,
                        args.batch_size,
                    )
                else:
                    train_data = train_activations_cache

                input_sensitivity, input_cond_sensitivity = compute_concept_attribution(
                    model,
                    args.layer_number,
                    test_image,
                    target_class,
                    get_grad_callback(select_method, train_data, probe, args.device),
                    sensitivity_method,
                    args.device,
                )
                save_path = os.path.join(
                    args.output_dir,
                    f"concept_attr_{sensitivity_method}_{select_method}_{image_name}.pdf",
                )
                visualize_concept_attribution(
                    test_image, args.concept, input_sensitivity, input_cond_sensitivity, sensitivity_method, save_path
                )

                evaluation_results[image_name][f"{select_method}:{sensitivity_method}"] = (
                    batch_evaluate_attribution_robustness(
                        model,
                        test_image,
                        input_cond_sensitivity,
                        probe,
                        args.layer_number,
                        target_class,
                        flip_percentages=[0, 5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100],
                        flip_method="mean",
                        device=args.device,
                    )
                )

    plot_save_path = os.path.join(args.output_dir, f"evaluation_{args.concept}_{args.layer_number}.pdf")
    plot_evaluation_results(evaluation_results, plot_save_path)


if __name__ == "__main__":
    main()
