"""
Utils fonctions for naming and organizing the training process.
"""

import os
import datetime
import json
import logging
from pathlib import Path


def create_model_output_dir():
    if not os.path.exists("outputs"):
        os.makedirs("outputs")
    base_dir = "outputs"
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_dir = os.path.join(base_dir, f"{timestamp}")
    os.makedirs(model_dir, exist_ok=True)
    subfolders = ["logs", "checkpoints"]
    for folder in subfolders:
        os.makedirs(os.path.join(model_dir, folder), exist_ok=True)
    return {
        "base": model_dir,
        "logs": os.path.join(model_dir, "logs"),
        "checkpoints": os.path.join(model_dir, "checkpoints"),
        "predictions": os.path.join(model_dir, "predictions"),
    }


def get_save_model_path(args, epoch):
    """Get the adequate model name to save it."""
    pretrain = True if args.model_unsupervised is not None else False
    save_name = f"{args.output_dir}/model_epoch_{epoch}_{Path(args.train_dataname).stem}_pretrained_{pretrain}.pt"
    return save_name


def get_save_classifier_path(args, epoch):
    """Get the adequate model name to save it."""
    pretrain = True if args.model_unsupervised is not None else False
    save_name = f"{args.output_dir}/model_epoch_{epoch}_{Path(args.train_dataname).stem}_pretrained_{pretrain}_classifier.pt"
    return save_name


def get_save_dnabert2_path(args, epoch):
    """Get the adequate model name to save it."""
    pretrain = True if args.model_unsupervised is not None else False
    save_name = f"{args.output_dir}/model_epoch_{epoch}_{Path(args.train_dataname).stem}_pretrained_{pretrain}_dnabert2.pt"
    return save_name


def save_args_to_json(args):
    """Save training parameters"""
    filename = "arguments.json"
    with open(Path(args.log_dir) / filename, "w", encoding="utf-8") as f:
        # Convert to dictionnary
        args = vars(args)
        json.dump(args, f, ensure_ascii=False, indent=4)
    logging.info("Training arguments saved in %s", filename)
