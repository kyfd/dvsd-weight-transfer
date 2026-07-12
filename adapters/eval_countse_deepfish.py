"""Evaluate the official CountSE checkpoint on the full DeepFish test split.

Protocol: zero-shot text query ``fish``, official FSC-147 test configuration,
official fixed thresholds, and no DeepFish fine-tuning or threshold selection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from groundingdino.util.utils import clean_state_dict
from main_inference import build_model_main, get_args_parser
from util.misc import nested_tensor_from_tensor_list
from util.slconfig import SLConfig
import datasets.transforms as T


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--bert", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("config/cfg_fsc147_test.py"))
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


def game_errors(gt_xy: np.ndarray, pred_xy: np.ndarray, height: int, width: int) -> dict[str, float]:
    result = {}
    for level in range(4):
        cells = 2**level
        gt_grid = np.zeros((cells, cells), dtype=np.float64)
        pred_grid = np.zeros((cells, cells), dtype=np.float64)
        for points, grid in ((gt_xy, gt_grid), (pred_xy, pred_grid)):
            if len(points) == 0:
                continue
            cols = np.minimum((points[:, 0] / max(width, 1) * cells).astype(int), cells - 1)
            rows = np.minimum((points[:, 1] / max(height, 1) * cells).astype(int), cells - 1)
            cols = np.maximum(cols, 0)
            rows = np.maximum(rows, 0)
            np.add.at(grid, (rows, cols), 1.0)
        result[f"GAME{level}"] = float(np.abs(gt_grid - pred_grid).sum())
    return result


def make_model_args(config_path: Path, bert_path: Path) -> argparse.Namespace:
    parser = get_args_parser()
    model_args = parser.parse_args(
        [
            "-c", str(config_path),
            "--datasets", "unused.json",
            "--output_dir", "unused",
            "--gpuid", "0",
            "--eval",
        ]
    )
    cfg = SLConfig.fromfile(str(config_path))
    cfg.merge_from_dict({"text_encoder_type": str(bert_path)})
    for key, value in cfg._cfg_dict.to_dict().items():
        if hasattr(model_args, key):
            raise RuntimeError(f"Configuration collides with parser argument: {key}")
        setattr(model_args, key, value)
    model_args.distributed = False
    model_args.rank = 0
    model_args.local_rank = 0
    model_args.amp = False
    return model_args


def transform_image(image: Image.Image) -> torch.Tensor:
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    tensor, _ = transform(image, None)
    return tensor


def atomic_write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def summarize(records: list[dict], metadata: dict, complete: bool, started: float) -> dict:
    abs_errors = np.asarray([row["absolute_error"] for row in records], dtype=np.float64)
    sq_errors = np.asarray([row["squared_error"] for row in records], dtype=np.float64)
    payload = {
        **metadata,
        "samples": len(records),
        "complete": complete,
        "limited_smoke_run": not complete,
        "mae": float(abs_errors.mean()) if len(records) else None,
        "rmse": float(math.sqrt(sq_errors.mean())) if len(records) else None,
        "game": {
            f"GAME{level}": float(np.mean([row["game"][f"GAME{level}"] for row in records]))
            if records else None
            for level in range(4)
        },
        "elapsed_seconds": time.time() - started,
        "per_image": records,
    }
    return payload


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    image_dir = args.data_root / "test" / "images"
    mask_dir = args.data_root / "test" / "masks"
    image_paths = sorted(image_dir.glob("*.jpg"))
    if len(image_paths) != args.expected_samples:
        raise RuntimeError(f"Expected {args.expected_samples} images, found {len(image_paths)}")

    torch.manual_seed(215)
    np.random.seed(215)
    model_args = make_model_args(args.config, args.bert)
    model, _, _ = build_model_main(model_args)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    incompat = model.load_state_dict(clean_state_dict(state), strict=False)
    # The released final checkpoint retains a training-only feature-map encoder
    # that the official FSC-147 inference graph does not instantiate.  The
    # official entry point also loads with strict=False; accept exactly those
    # surplus keys while keeping every inference-graph mismatch fatal.
    unexpected = [
        key for key in incompat.unexpected_keys
        if not key.startswith("feature_map_encoder.")
    ]
    if incompat.missing_keys or unexpected:
        raise RuntimeError(
            f"Checkpoint mismatch: missing={incompat.missing_keys}, unexpected={unexpected}"
        )
    model.cuda().eval()

    metadata = {
        "method": "CountSE",
        "venue": "ICCV 2025 Highlight",
        "protocol": (
            "official pretrained checkpoint; zero-shot text-only DeepFish test; "
            "prompt fish; FSC-147 test preprocessing; fixed box threshold 0.35 and text threshold 0; "
            "no DeepFish fine-tuning or threshold selection"
        ),
        "split": "test",
        "prompt": args.prompt,
        "box_threshold": float(model_args.box_threshold),
        "text_threshold": float(model_args.text_threshold),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256(args.checkpoint),
        "official_checkpoint_epoch": checkpoint.get("epoch") if isinstance(checkpoint, dict) else None,
        "repo_commit": "fcc82a91495df3ebf2d16ab64643dc358e85e8c4",
    }
    records: list[dict] = []
    if args.output.exists():
        previous = json.loads(args.output.read_text(encoding="utf-8"))
        records = previous.get("per_image", [])
    completed_names = {row["name"] for row in records}
    started = time.time()

    caption = f"{args.prompt} ."
    for image_path in image_paths:
        if image_path.name in completed_names:
            continue
        mask_path = mask_dir / f"{image_path.stem}.png"
        if not mask_path.exists():
            raise FileNotFoundError(mask_path)
        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        gt_yx = np.argwhere(np.asarray(Image.open(mask_path)) > 0)
        gt_xy = gt_yx[:, [1, 0]].astype(np.float64) if len(gt_yx) else np.zeros((0, 2))

        sample = transform_image(image).cuda()
        nested = nested_tensor_from_tensor_list([sample])
        outputs = model(
            nested,
            [torch.tensor([0], device="cuda")],
            [caption],
            captions=[caption],
        )
        token_ids = outputs["token"]["input_ids"][0]
        period_locations = (token_ids == 1012).nonzero(as_tuple=False)
        if len(period_locations) == 0:
            raise RuntimeError(f"Prompt tokenization lacks period token for {image_path.name}")
        end_idx = int(period_locations[0].item())
        logits = outputs["pred_logits"].sigmoid()[0]
        boxes = outputs["pred_boxes"][0]
        box_mask = logits.max(dim=-1).values > model_args.box_threshold
        logits, boxes = logits[box_mask], boxes[box_mask]
        text_mask = (logits[:, 1:end_idx] > model_args.text_threshold).sum(dim=-1) == (end_idx - 1)
        boxes = boxes[text_mask]
        if len(boxes):
            centres = boxes[:, :2].detach().cpu().numpy()
            pred_xy = centres * np.asarray([width, height], dtype=np.float64)
        else:
            pred_xy = np.zeros((0, 2), dtype=np.float64)

        prediction = float(len(pred_xy))
        ground_truth = float(len(gt_xy))
        difference = prediction - ground_truth
        records.append(
            {
                "name": image_path.name,
                "ground_truth": ground_truth,
                "prediction": prediction,
                "absolute_error": abs(difference),
                "squared_error": difference * difference,
                "game": game_errors(gt_xy, pred_xy, height, width),
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
