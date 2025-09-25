"""
Concept probing visualization for animal images and dataset samples.

Run with:

```
uv run --group scripts -m scripts.composition.concept_probing
```
"""

import argparse
import random
import os
import torch
import timm
import matplotlib.pyplot as plt
from loguru import logger
from datasets import load_dataset
from tensordict import TensorDict

from tdhook.latent.activation_caching import ActivationCaching

from .concept_utils import load_and_preprocess_image, create_balanced_dataset, collect_activations, train_probe


def get_probe_probability(model, probe, image, layer_number, device="cpu"):
    """Get probe probability for a single image."""
    with torch.no_grad():
        with ActivationCaching(f"features.{layer_number}").prepare(model) as hooked_model:
            input_td = TensorDict({"input": image.unsqueeze(0).to(device)}, batch_size=1)
            hooked_model(input_td)
            cache = hooked_model.hooking_context.cache
            activations = cache[f"features.{layer_number}"].amax(dim=(2, 3))

        flattened_activations = activations.view(1, -1).cpu().numpy()
        proba = probe.predict_proba(flattened_activations)[0]

    return proba[1]


def create_visualization_figure(images, probabilities, concept, figure_type, save_path=None):
    """Create a single visualization figure for a group of images."""
    n_images = len(images)
    fig, axes = plt.subplots(1, n_images, figsize=(4 * n_images, 4))

    if n_images == 1:
        axes = [axes]

    for i, (image_name, image_tensor) in enumerate(images.items()):
        image_np = image_tensor.permute(1, 2, 0).cpu().numpy()
        image_np = (image_np - image_np.min()) / (image_np.max() - image_np.min())

        axes[i].imshow(image_np)
        axes[i].set_title(f"{image_name}\nProbability: {probabilities[image_name]:.3f}")
        axes[i].axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Visualization saved to {save_path}")

    return fig


def create_animal_figures(animal_images, probe_probabilities, concept, output_dir, method):
    """Create separate figures for animal groups."""
    animals_1 = {k: v for k, v in animal_images.items() if k.endswith("_1")}
    animals_2 = {k: v for k, v in animal_images.items() if k.endswith("_2")}

    if animals_1:
        prob_1 = {k: v for k, v in probe_probabilities.items() if k.endswith("_1")}
        fig1 = create_visualization_figure(
            animals_1,
            prob_1,
            concept,
            "Animals Group 1",
            os.path.join(output_dir, f"probe_{method}_animals_group_1.pdf"),
        )
        plt.close(fig1)

    if animals_2:
        prob_2 = {k: v for k, v in probe_probabilities.items() if k.endswith("_2")}
        fig2 = create_visualization_figure(
            animals_2,
            prob_2,
            concept,
            "Animals Group 2",
            os.path.join(output_dir, f"probe_{method}_animals_group_2.pdf"),
        )
        plt.close(fig2)


def create_dataset_figures(dataset_images, dataset_probabilities, concept, output_dir, method):
    """Create separate figures for positive and negative dataset samples."""
    positive_images = {k: v for k, v in dataset_images.items() if k.startswith("positive_")}
    negative_images = {k: v for k, v in dataset_images.items() if k.startswith("negative_")}

    if positive_images:
        pos_prob = {k: v for k, v in dataset_probabilities.items() if k.startswith("positive_")}
        fig_pos = create_visualization_figure(
            positive_images,
            pos_prob,
            concept,
            "Positive Samples",
            os.path.join(output_dir, f"probe_{method}_positive_samples.pdf"),
        )
        plt.close(fig_pos)

    if negative_images:
        neg_prob = {k: v for k, v in dataset_probabilities.items() if k.startswith("negative_")}
        fig_neg = create_visualization_figure(
            negative_images,
            neg_prob,
            concept,
            "Negative Samples",
            os.path.join(output_dir, f"probe_{method}_negative_samples.pdf"),
        )
        plt.close(fig_neg)


def main():
    """Main function to create probe visualizations."""
    parser = argparse.ArgumentParser(description="Concept Probing Visualization")
    parser.add_argument("--layer_number", type=int, default=28, help="Layer number to analyze")
    parser.add_argument("--concept", default="striped", help="Concept to analyze")
    parser.add_argument(
        "--method",
        default="mean_difference",
        choices=["logistic_regression_l1", "logistic_regression_l2", "mean_difference"],
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

    logger.info("Loading VGG16 model...")
    model = timm.create_model("vgg16.tv_in1k", pretrained=True)
    logger.info("Replacing ReLU with ReLU(inplace=False)...")
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
        model, args.layer_number, train_concept_dataset, args.device, batch_size=args.batch_size
    )
    probe = train_probe(train_activations_cache, args.concept, args.method)

    animal_images = {}
    image_paths = {
        "zebra_1": "results/concept_attribution/zebra_1.jpg",
        "zebra_2": "results/concept_attribution/zebra_2.jpg",
        "lemur_1": "results/concept_attribution/lemur_1.jpg",
        "lemur_2": "results/concept_attribution/lemur_2.jpg",
        "skunk_1": "results/concept_attribution/skunk_1.jpg",
        "skunk_2": "results/concept_attribution/skunk_2.jpg",
    }

    logger.info("Loading animal images...")
    for name, path in image_paths.items():
        if os.path.exists(path):
            image_tensor = load_and_preprocess_image(path, transforms, args.device)
            animal_images[name] = image_tensor
        else:
            logger.warning(f"Image not found: {path}")

    probe_probabilities = {}
    logger.info("Computing probe probabilities...")
    for name, image_tensor in animal_images.items():
        proba = get_probe_probability(model, probe, image_tensor, args.layer_number, args.device)
        probe_probabilities[name] = proba
        logger.info(f"{name}: probe probability = {proba:.3f}")

    os.makedirs(args.output_dir, exist_ok=True)

    logger.info("Testing with dataset examples...")

    positive_indices = random.sample([i for i, item in enumerate(test_dataset) if item["label"] == concept_index], 3)
    negative_indices = random.sample([i for i, item in enumerate(test_dataset) if item["label"] != concept_index], 3)

    dataset_images = {}
    dataset_probabilities = {}

    for i, idx in enumerate(positive_indices):
        image = test_dataset[idx]["image"]
        image_tensor = transforms(image)
        image_tensor = image_tensor.to(args.device)
        dataset_images[f"positive_{i}"] = image_tensor
        proba = get_probe_probability(model, probe, image_tensor, args.layer_number, args.device)
        dataset_probabilities[f"positive_{i}"] = proba
        logger.info(f"Positive example {i}: probe probability = {proba:.3f}")

    for i, idx in enumerate(negative_indices):
        image = test_dataset[idx]["image"]
        image_tensor = transforms(image)
        image_tensor = image_tensor.to(args.device)
        dataset_images[f"negative_{i}"] = image_tensor
        proba = get_probe_probability(model, probe, image_tensor, args.layer_number, args.device)
        dataset_probabilities[f"negative_{i}"] = proba
        logger.info(f"Negative example {i}: probe probability = {proba:.3f}")

    logger.info("Creating modular visualizations...")
    create_animal_figures(animal_images, probe_probabilities, args.concept, args.output_dir, args.method)
    create_dataset_figures(dataset_images, dataset_probabilities, args.concept, args.output_dir, args.method)

    logger.info("Visualization complete!")


if __name__ == "__main__":
    main()
