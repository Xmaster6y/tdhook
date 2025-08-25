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
from lczerolens.concepts import HasThreat
from lczerolens.data import BoardData

from tdhook.latent.probing import Probing, SklearnProbeManager


def main(args: argparse.Namespace):
    """Run concept probing."""
    logger.info("Starting concept probing...")

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

    probe_manager = SklearnProbeManager(LogisticRegression, {}, lambda x, y: concept.compute_metrics(x, y))

    with Probing("block\d/conv2/relu", probe_manager.probe_factory, additional_keys=["labels", "step_type"]).prepare(
        model
    ) as hooked_module:
        with torch.no_grad():
            train_inputs = TensorDict(
                {
                    "boards": model.prepare_boards(*train_boards),
                    "labels": torch.tensor(train_labels),
                    "step_type": "fit",
                },
                batch_size=len(train_boards),
            )
            hooked_module(train_inputs)

            test_inputs = TensorDict(
                {
                    "boards": model.prepare_boards(*test_boards),
                    "labels": torch.tensor(test_labels),
                    "step_type": "predict",
                },
                batch_size=len(test_boards),
            )
            hooked_module(test_inputs)

    for key, value in probe_manager.metrics.items():
        logger.info(f"{key}: {value}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("concept-probing")
    parser.add_argument("--model_path", type=str, default="results/lczerolens/maia-1900.onnx")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
