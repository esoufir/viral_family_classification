"""
Main code for training the classifier.
"""

import os
import argparse
import datetime
import logging
from utils import create_model_output_dir, save_args_to_json
from dataloader import contig_loader_csv
from models import Trainer


logger = logging.getLogger(__name__)


def main(args):
    # Initializing training directories & logging
    utils_dir = create_model_output_dir()
    logging.basicConfig(
        filename=f"{utils_dir['logs']}/classification_dnabert2.log",
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger.info("Starting finetuning of DNABERT2")
    args.output_dir = utils_dir["checkpoints"]
    args.log_dir = utils_dir["logs"]
    save_args_to_json(args)
    # Loading datasets :
    train_loader = contig_loader_csv(args, load_train=True)
    val_loader = contig_loader_csv(args, load_train=False)
    all_labels = []
    for batch in train_loader:
        all_labels.extend(batch["label"].tolist())
    args.num_classes = len(set(all_labels))
    # Main training :
    trainer = Trainer(args)
    if args.resume:
        trainer.resume(train_loader, val_loader, args)
    else:
        trainer.training_loop(train_loader, val_loader, args)


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    parser = argparse.ArgumentParser()
    parser.add_argument("--datapath", type=str, help="Path to data folder.")
    parser.add_argument("--train_dataname", type=str, help="Training data filename.")
    parser.add_argument("--val_dataname", type=str, help="Validation data filename.")
    parser.add_argument("--train_batch_size", type=int, default=64, help="Train batch size.")
    parser.add_argument("--val_batch_size", type=int, default=64, help="Validation batch size.")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs for training.")
    parser.add_argument("--start_epoch", type=int, default=1, help="Init epoch to start the training.")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate value.")
    parser.add_argument(
        "--max_length", type=int, default=300, help="Max length of tokens"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=f"./checkpoints_{str(datetime.datetime.now()).replace(' ', '')}",
        help="Directory to save model checkpoints",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="DNABERT_2_pretrained",
        help="Path to used tokenizer",
    )
    parser.add_argument(
        "--model_unsupervised",
        type=str,
        default="DNABERT_2_pretrained",
        help="Path to used pretrained weights.",
    )
    parser.add_argument("--label_column", type=str, default="id_Family")
    parser.add_argument("--subsample_rank", type=str, default=None)
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--subsample", action="store_true")
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume the training using the given checkpoint.",
    )
    args = parser.parse_args()
    main(args)
