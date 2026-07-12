"""Evaluate an official-code LGCount retraining on the full DeepFish test set."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

# The released entry point imports Gradio although its training and test paths
# never reference it.  Keep the inference environment minimal without changing
# any counting code.
if importlib.util.find_spec("gradio") is None:
    sys.modules["gradio"] = types.ModuleType("gradio")

import util.misc as misc
from run import Model, get_args_parser
from util.constant import SCALE_FACTOR


COARSE = ["zero", "one hundred", "two hundred", "five hundred", "one thousand", "infinity"]
FINE = [
    ["zero", "twenty", "forty", "sixty", "eighty", "one hundred"],
    ["one hundred", "one hundred twenty", "one hundred forty", "one hundred sixty", "one hundred eighty", "two hundred"],
    ["two hundred", "two hundred fifty", "three hundred", "three hundred fifty", "four hundred", "five hundred"],
    ["five hundred", "six hundred", "seven hundred", "eight hundred", "nine hundred", "one thousand"],
    ["one thousand", "one thousand two hundred", "one thousand four hundred", "one thousand eight hundred", "two thousand five hundred", "infinity"],
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--alignment-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt", default="fish")
    parser.add_argument("--expected-samples", type=int, default=960)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prompt_lists(noun: str) -> tuple[list[str], list[list[str]]]:
    coarse = [f"There are between {COARSE[i]} and {COARSE[i + 1]} {noun}" for i in range(5)]
    fine = [
        [f"There are between {FINE[i][j]} and {FINE[i][j + 1]} {noun}" for j in range(5)]
        for i in range(5)
    ]
    return coarse, fine


def game_errors(prediction: np.ndarray, gt_xy: np.ndarray) -> dict[str, float]:
    height, width = prediction.shape
    result = {}
    for level in range(4):
        cells = 2**level
        gt_grid = np.zeros((cells, cells), dtype=np.float64)
        if len(gt_xy):
            cols = np.minimum((gt_xy[:, 0] / max(width, 1) * cells).astype(int), cells - 1)
            rows = np.minimum((gt_xy[:, 1] / max(height, 1) * cells).astype(int), cells - 1)
            np.add.at(gt_grid, (np.maximum(rows, 0), np.maximum(cols, 0)), 1.0)
        y_edges = np.linspace(0, height, cells + 1, dtype=int)
        x_edges = np.linspace(0, width, cells + 1, dtype=int)
        error = 0.0
        for row in range(cells):
            for col in range(cells):
                pred_count = prediction[y_edges[row]:y_edges[row + 1], x_edges[col]:x_edges[col + 1]].sum()
                error += abs(float(pred_count) - float(gt_grid[row, col]))
        result[f"GAME{level}"] = float(error)
    return result


def atomic_write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def summarize(records: list[dict], metadata: dict, complete: bool, started: float) -> dict:
    absolute = np.asarray([row["absolute_error"] for row in records], dtype=np.float64)
    squared = np.asarray([row["squared_error"] for row in records], dtype=np.float64)
    return {
        **metadata,
        "samples": len(records),
        "complete": complete,
        "limited_smoke_run": not complete,
        "mae": float(absolute.mean()) if len(records) else None,
        "rmse": float(math.sqrt(squared.mean())) if len(records) else None,
        "game": {
            f"GAME{level}": float(np.mean([row["game"][f"GAME{level}"] for row in records]))
            if records else None
            for level in range(4)
        },
        "elapsed_seconds": time.time() - started,
        "per_image": records,
    }


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    official_args = get_args_parser().parse_args([])
    official_args.mode = "test"
    official_args.dataset_type = "FSC"
    official_args.ckpt = str(args.checkpoint)

    expected_alignment = Path("ckpt/epoch=149-avg_fine_accuracy_pred=0.71.ckpt")
    expected_alignment.parent.mkdir(parents=True, exist_ok=True)
    if not expected_alignment.exists():
        try:
            expected_alignment.symlink_to(args.alignment_checkpoint.resolve())
        except OSError:
            import shutil
            shutil.copy2(args.alignment_checkpoint, expected_alignment)

    model = Model(official_args)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
    incompat = model.load_state_dict(state, strict=False)
    allowed_missing = tuple(key for key in incompat.missing_keys if ".clip." in key or ".vit." in key)
    disallowed_missing = [key for key in incompat.missing_keys if key not in allowed_missing]
    if disallowed_missing or incompat.unexpected_keys:
        raise RuntimeError(
            f"Checkpoint mismatch: missing={disallowed_missing}, unexpected={incompat.unexpected_keys}"
        )
    model.cuda().eval()

    image_dir = args.data_root / "test" / "images"
    mask_dir = args.data_root / "test" / "masks"
    image_paths = sorted(image_dir.glob("*.jpg"))
    if len(image_paths) != args.expected_samples:
        raise RuntimeError(f"Expected {args.expected_samples} images, found {len(image_paths)}")

    metadata = {
        "method": "LGCount",
        "venue": "ICCV 2025",
        "protocol": (
            "official-code FSC-147 retraining; best validation checkpoint; zero-shot DeepFish test; "
            "prompt fish; official 384-pixel-height sliding-window inference; no DeepFish fine-tuning"
        ),
        "split": "test",
        "prompt": args.prompt,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256(args.checkpoint),
        "alignment_checkpoint_sha256": sha256(args.alignment_checkpoint),
        "repo_commit": "c746346dee137cc65a0ede4155e4837a74c08868",
        "scale_factor": SCALE_FACTOR,
    }
    records: list[dict] = []
    if args.output.exists():
        previous = json.loads(args.output.read_text(encoding="utf-8"))
        records = previous.get("per_image", [])
    completed = {row["name"] for row in records}
    started = time.time()
    coarse, fine = prompt_lists(args.prompt)

    preprocess = transforms.Compose([transforms.Resize(384), transforms.ToTensor()])
    for image_path in image_paths:
        if image_path.name in completed:
            continue
        mask_path = mask_dir / f"{image_path.stem}.png"
        image_pil = Image.open(image_path).convert("RGB")
        original_width, original_height = image_pil.size
        image = preprocess(image_pil)
        raw_height, raw_width = image.shape[-2:]
        if raw_height != 384:
            raise RuntimeError(f"Unexpected resized height {raw_height} for {image_path.name}")

        patches_np, _ = misc.sliding_window(image.unsqueeze(0), stride=128)
        patches = torch.from_numpy(patches_np).float().cuda()
        class_text = np.asarray([args.prompt] * len(patches))
        coarse_batch = [[text] * len(patches) for text in coarse]
        fine_batch = [[[text] * len(patches) for text in row] for row in fine]
        _, _, top_fine_embedding = model.model_align(patches, coarse_batch, fine_batch)
        output, _ = model.model(patches, class_text, 220, top_fine_embedding)
        output = misc.window_composite(output.unsqueeze(1), stride=128).squeeze(1)
        output = output[:, :, :raw_width]
        density = output[0].detach().cpu().numpy().astype(np.float64) / SCALE_FACTOR

        gt_yx = np.argwhere(np.asarray(Image.open(mask_path)) > 0)
        if len(gt_yx):
            gt_xy = gt_yx[:, [1, 0]].astype(np.float64)
            gt_xy[:, 0] *= raw_width / original_width
            gt_xy[:, 1] *= raw_height / original_height
        else:
            gt_xy = np.zeros((0, 2), dtype=np.float64)
        prediction = float(density.sum())
        ground_truth = float(len(gt_xy))
        difference = prediction - ground_truth
        records.append(
            {
                "name": image_path.name,
                "ground_truth": ground_truth,
                "prediction": prediction,
                "absolute_error": abs(difference),
                "squared_error": difference * difference,
                "resized_shape": [raw_height, raw_width],
                "game": game_errors(density, gt_xy),
            }
        )
        if len(records) % args.save_every == 0:
            atomic_write(args.output, summarize(records, metadata, False, started))
            print(f"[{len(records)}/{len(image_paths)}]", flush=True)
        if args.limit and len(records) >= args.limit:
            break

    complete = len(records) == args.expected_samples and not args.limit
    payload = summarize(records, metadata, complete, started)
    atomic_write(args.output, payload)
    print(json.dumps({key: value for key, value in payload.items() if key != "per_image"}, indent=2))


if __name__ == "__main__":
    main()
