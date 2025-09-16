"""
Script to compute value saliency in a chess model.

Run with:

```
uv run --group scripts -m scripts.lczerolens.value_saliency
```
"""

import argparse
from loguru import logger
from tensordict import TensorDict

from lczerolens import LczeroModel, LczeroBoard

from tdhook.attribution.gradient_helpers import Saliency


def main(args: argparse.Namespace):
    """Run all benchmarks."""
    logger.info("Starting lczerolens value saliency...")

    model = LczeroModel.from_path(args.model_path)
    board = LczeroBoard(args.fen)
    for move in args.moves.split(" "):
        board.push_uci(move)
    td = TensorDict(board=model.prepare_boards(board), batch_size=1)

    def best_logit_init_targets(td: TensorDict, _):
        policy = td["policy"]
        best_logit = policy.max(dim=-1).values
        return TensorDict(out=best_logit, batch_size=td.batch_size)

    logger.info("Computing best move saliency...")
    saliency_context = Saliency(init_attr_targets=best_logit_init_targets)
    with saliency_context.prepare(model) as hooked_model:
        output = hooked_model(td)
        move = board.decode_move(output[("_mod_out", "policy")][0].argmax())
        arrows = [(move.from_square, move.to_square)]
        logger.info(f"Best move: ({move})")
        attr = output.get(("attr", "board")).sum(dim=1).view(64)
        board.render_heatmap(attr, arrows=arrows, save_to="results/lczerolens/best_move_saliency.svg", normalise="abs")

    def get_init_targets(idx: int):
        def init_targets(td, _):
            return TensorDict(out=td["wdl"][..., idx], batch_size=td.batch_size)

        return init_targets

    for idx, name in enumerate(["win", "draw", "lose"]):
        logger.info(f"Computing {name} saliency...")
        saliency_context = Saliency(init_attr_targets=get_init_targets(idx))
        with saliency_context.prepare(model) as hooked_model:
            output = hooked_model(td)
            logger.info(f"{name} output: {output[('_mod_out', 'wdl')][0, idx]:.2f}")
            attr = output.get(("attr", "board")).sum(dim=1).view(64)
            board.render_heatmap(attr, save_to=f"results/lczerolens/{name}_saliency.svg", normalise="abs")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("value-saliency")
    parser.add_argument("--model_path", type=str, default="results/lczerolens/maia-1900.onnx")
    parser.add_argument("--fen", type=str, default="5k2/2R5/1PQ5/2Pp1n2/5P2/2b1r3/3K2P1/8 w - - 11 42")
    parser.add_argument("--moves", type=str, default="d2c2 f5d4 c2b1")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
