"""Restartable LGCount training using the official FSC-147 configuration."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import types
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning import Trainer, seed_everything

# The official entry point imports Gradio but never uses it in training or
# evaluation.  Avoid installing the unrelated web UI dependency on the compute
# node while leaving all model and optimization code unchanged.
if importlib.util.find_spec("gradio") is None:
    sys.modules["gradio"] = types.ModuleType("gradio")

from run import Model, get_args_parser
from util.FSC147 import FSC147


def _compat_lr_scheduler_step(self, scheduler, optimizer_idx, metric) -> None:
    """Preserve the official StepLR behavior under PyTorch 2.x/PL 1.8."""
    scheduler.step()


# PyTorch 2.x moved StepLR to the public LRScheduler base class, while
# Lightning 1.8 checks the removed private base class.  Declaring the standard
# hook bypasses only that stale type check and keeps one scheduler step per
# epoch, exactly as in the released training code.
Model.lr_scheduler_step = _compat_lr_scheduler_step


def parse_args() -> Tuple[argparse.Namespace, Optional[Path], Path]:
    parser = get_args_parser()
    parser.add_argument("--resume-latest", type=Path, default=None)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    return args, args.resume_latest, args.summary


def main() -> None:
    args, resume_latest, summary_path = parse_args()
    seed_everything(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    train_set = FSC147(split="train")
    val_set = FSC147(split="val", resize_val=False)
    train_loader = torch.utils.data.DataLoader(
        train_set,
        sampler=torch.utils.data.RandomSampler(train_set),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )
    val_loader = torch.utils.data.DataLoader(
        val_set,
        sampler=torch.utils.data.SequentialSampler(val_set),
        batch_size=1,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )

    checkpoint_dir = Path(args.output_dir) / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    callback = pl.callbacks.ModelCheckpoint(
        dirpath=str(checkpoint_dir),
        monitor="val_mae",
        save_top_k=4,
        save_last=True,
        mode="min",
        filename="{epoch}-{val_mae:.4f}",
        every_n_epochs=1,
    )
    logger = pl.loggers.TensorBoardLogger(args.output_dir, name=args.exp_name)
    model = Model(args, all_classes=train_set.all_classes)
    trainer = Trainer(
        accelerator="gpu",
        devices=1,
        callbacks=[callback],
        accumulate_grad_batches=args.accum_iter,
        precision=16,
        max_epochs=args.epochs + args.contrast_pre_epoch,
        logger=logger,
        check_val_every_n_epoch=args.val_freq,
    )
    ckpt_path = None
    if resume_latest is not None and resume_latest.is_file():
        ckpt_path = str(resume_latest.resolve())
    trainer.fit(model, train_loader, val_loader, ckpt_path=ckpt_path)

    summary = {
        "method": "LGCount",
        "repo_commit": "c746346dee137cc65a0ede4155e4837a74c08868",
        "training_data": "FSC-147 official train split",
        "seed": args.seed,
        "epochs": args.epochs + args.contrast_pre_epoch,
        "start_val_epoch": args.start_val_epoch,
        "best_model_path": callback.best_model_path,
        "best_model_score": float(callback.best_model_score)
        if callback.best_model_score is not None else None,
        "last_model_path": callback.last_model_path,
        "complete": trainer.current_epoch >= args.epochs + args.contrast_pre_epoch,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = summary_path.with_suffix(summary_path.suffix + ".tmp")
    temporary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    os.replace(temporary, summary_path)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
