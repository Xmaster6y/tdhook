"""
Utils for concept probing and attribution.
"""

import random
import torch
import numpy as np
from PIL import Image
from loguru import logger
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
from datasets import Dataset
from tensordict import TensorDict

from tdhook.latent.activation_caching import ActivationCaching
from tdhook.latent.probing import MeanDifferenceClassifier


def load_and_preprocess_image(image_path, transforms, device="cpu"):
    logger.info(f"Loading custom image from: {image_path}")

    image = Image.open(image_path).convert("RGB")
    image_tensor = transforms(image)

    logger.info(f"Image loaded and preprocessed. Shape: {image_tensor.shape}")
    return image_tensor.to(device)


def create_balanced_dataset(dataset, concept_index, concept_name, transforms, num_samples_per_class=None, seed=42):
    random.seed(seed)
    np.random.seed(seed)

    positive_indices = [i for i, item in enumerate(dataset) if item["label"] == concept_index]
    logger.info(f"Found {len(positive_indices)} positive examples for concept '{concept_name}'")

    negative_indices = [i for i, item in enumerate(dataset) if item["label"] != concept_index]
    logger.info(f"Found {len(negative_indices)} negative examples")

    if num_samples_per_class is None:
        num_samples_per_class = min(len(positive_indices), len(negative_indices))
    else:
        num_samples_per_class = min(num_samples_per_class, len(positive_indices), len(negative_indices))

    positive_indices = random.sample(positive_indices, num_samples_per_class)
    negative_indices = random.sample(negative_indices, num_samples_per_class)

    indices = positive_indices + negative_indices
    labels = [1] * len(positive_indices) + [0] * len(negative_indices)

    combined = list(zip(indices, labels))
    random.shuffle(combined)
    indices, labels = zip(*combined)

    transformed_images = []
    for i in indices:
        image = dataset[i]["image"]
        if transforms:
            image = transforms(image)
        transformed_images.append(image)

    balanced_data = {
        "image": transformed_images,
        "label": list(labels),
        "original_label": [dataset[i]["label"] for i in indices],
    }

    hf_dataset = Dataset.from_dict(balanced_data).with_format("torch")

    logger.info(
        f"Created balanced dataset with {len(positive_indices)} positive and {len(negative_indices)} negative examples"
    )

    return hf_dataset


def collect_activations(model, layer_number, concept_dataset, device="cuda", batch_size=32):
    logger.info(f"Collecting activations from features.{layer_number} with batch size {batch_size}...")

    dataloader = DataLoader(
        concept_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    activations_list = []
    labels_list = []

    with ActivationCaching(f"features.{layer_number}").prepare(model) as hooked_model:
        for batch_idx, batch in enumerate(dataloader):
            batch_tensor = batch["image"].to(device)
            batch_labels = batch["label"]

            input_td = TensorDict({"input": batch_tensor}, batch_size=[])
            with torch.no_grad():
                hooked_model(input_td)

            cache = hooked_model.hooking_context.cache
            batch_activations = cache[f"features.{layer_number}"].amax(dim=(2, 3))

            activations_list.append(batch_activations.cpu())
            labels_list.append(batch_labels.cpu())

            logger.info(f"Processed batch {batch_idx + 1}/{len(dataloader)}")

    activations_cache = TensorDict(
        {
            "activations": torch.cat(activations_list, dim=0),
            "labels": torch.cat(labels_list, dim=0),
        }
    )

    logger.info(f"Collected {len(activations_list)} activations with shape {activations_list[0].shape}")
    return activations_cache


def train_probe(activations_cache, concept, method="mean_difference"):
    logger.info(f"Training probe for concept: {concept} using {method}")

    activations = activations_cache["activations"]
    labels = activations_cache["labels"]

    batch_size = activations.shape[0]
    flattened_activations = activations.view(batch_size, -1).numpy()

    concept_labels = np.array(labels, dtype=int)

    if np.sum(concept_labels) == 0:
        raise ValueError(f"No positive examples found for concept: {concept}")

    logger.info(f"Found {np.sum(concept_labels)} positive examples out of {len(concept_labels)} total")

    if method == "logistic_regression_l1":
        probe = LogisticRegression(penalty="l1", random_state=42, max_iter=1000, solver="liblinear")
    elif method == "logistic_regression_l2":
        probe = LogisticRegression(penalty="l2", random_state=42, max_iter=1000, solver="liblinear")
    elif method == "mean_difference":
        probe = MeanDifferenceClassifier()
    else:
        raise ValueError(
            f"Unknown method: {method}. Choose 'logistic_regression_l1', 'logistic_regression_l2', or 'mean_difference'"
        )

    probe.fit(flattened_activations, concept_labels)

    predictions = probe.predict(flattened_activations)
    accuracy = accuracy_score(concept_labels, predictions)
    logger.info(f"CAV training accuracy for {concept}: {accuracy:.3f}")

    return probe
