"""
Script to probe concepts in a chess model.

Run with:

```
uv run --group scripts -m scripts.lczerolens.concept_probing
```
"""

import argparse
from loguru import logger
from tensordict import TensorDict
import torch

from datasets import load_dataset, Dataset
from sklearn.linear_model import LogisticRegression
from lczerolens import LczeroModel
from lczerolens.concepts import Concept, HasThreat
from lczerolens.data import BoardData

from tdhook.acteng import ActivationCaching


def train_logistic_regression(cache: TensorDict, labels: TensorDict):
    """Train a logistic regression model."""
    models = {}
    for key, value in cache.items():
        model = LogisticRegression()
        model.fit(value.flatten(1), labels)
        models[key] = model
    return models


def test_logistic_regression(cache: TensorDict, labels: TensorDict, models: dict, concept: Concept):
    """Test a logistic regression model."""
    metrics = {}
    for key, value in cache.items():
        model = models[key]
        predictions = model.predict(value.flatten(1))
        metrics[key] = concept.compute_metrics(predictions, labels)
    return metrics


def main(args: argparse.Namespace):
    """Run all benchmarks."""
    logger.info("Starting test stats...")

    model = LczeroModel.from_path(args.model_path)
    dataset = load_dataset("lczerolens/tcec-boards", split="train", streaming=True)
    work_ds = dataset.shuffle(seed=42).take(1000)
    work_ds = Dataset.from_generator(lambda: (yield from work_ds), features=work_ds.features)
    work_ds = work_ds.train_test_split(test_size=0.2)
    train_ds = work_ds["train"]
    test_ds = work_ds["test"]

    concept = HasThreat("K")
    train_boards, train_labels = BoardData.concept_collate_fn(list(train_ds), concept)
    test_boards, test_labels = BoardData.concept_collate_fn(list(test_ds), concept)

    logger.info(f"Positive examples (train): {sum(train_labels)}")
    logger.info(f"Positive examples (test): {sum(test_labels)}")

    train_cache = TensorDict(batch_size=len(train_boards))
    test_cache = TensorDict(batch_size=len(test_boards))

    with ActivationCaching("block\d/conv2/relu", cache=train_cache).prepare(model) as hooked_module:
        with torch.no_grad():
            hooked_module(train_boards)

    with ActivationCaching("block\d/conv2/relu", cache=test_cache).prepare(model) as hooked_module:
        with torch.no_grad():
            hooked_module(test_boards)

    train_models = train_logistic_regression(train_cache, train_labels)
    metrics = test_logistic_regression(test_cache, test_labels, train_models, concept)
    for key, value in metrics.items():
        logger.info(f"{key}: {value}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("concept-probing")
    parser.add_argument("--model_path", type=str, default="results/lczerolens/maia-1900.onnx")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
