"""
Model definition and training code.
"""

import os
import logging
from collections import OrderedDict
from tqdm import tqdm
import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel
from utils import (
    get_save_model_path,
    save_args_to_json,
    get_save_dnabert2_path,
    get_save_classifier_path,
)

logger = logging.getLogger(__name__)

# MODELS
class DnaBert2(nn.Module):
    def __init__(self, args):
        """Load pretrained DNABERT2 weights"""
        super().__init__()
        logger.info("Initializing DNABert_2")
        # Load pre-existing weights
        self.dnabert2 = AutoModel.from_pretrained(
            args.model_unsupervised, trust_remote_code=True, revision="main"
        )
        if args.freeze:
            # Freeze all bert layers to train only classification head :
            logger.info("Freeze DNABert_2 layers")
            for param in self.dnabert2.parameters():
                param.requires_grad = False
        else:
            logger.info("Weights are not frozen. All weights are trainables.")

    def forward(self, input_ids, attention_mask):
        """Passes forward the encoder and returns mean embedding representation."""
        hidden_states = self.dnabert2(
            input_ids=input_ids, attention_mask=attention_mask
        )[0]
        embedding_mean = torch.mean(hidden_states, dim=1)
        return embedding_mean


class Classifier(nn.Module):
    """Classifier to perform taxonomic classification"""

    def __init__(self, args):
        super().__init__()
        # Encoder parameters :
        self.dnabert_2 = DnaBert2(args)
        self.input_shape = 768
        # Number of viral families :
        self.num_classes = args.num_classes
        # Classification head :
        self.classifier = nn.Sequential(
            nn.Linear(self.input_shape, 1024),
            nn.ReLU(),
            nn.LayerNorm(1024),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.LayerNorm(512),
            nn.Dropout(0.3),
            nn.Linear(512, self.num_classes),
        )

    def forward(self, input_ids, attention_mask):
        """Passes forward the encoder and the classification head
        for predictions."""
        # Through the encoder :
        output = self.dnabert_2(input_ids, attention_mask)
        # Through the classification head:
        x = self.classifier(output)
        return x


def count_parameters(model):
    """Return parameters count."""
    total = 0
    print("\n[Trainable parameters :]")
    for name, param in model.named_parameters():
        if param.requires_grad:
            param_count = param.numel()
            print(f"{name:60} | {param_count:>10,} parameters")
            total += param_count
    print(f"\nTotal number of trainable parameters : {total:,}\n")
    return total


class Trainer:
    """
    Object for training the classifier.

    ...

    Attributes
    ----------
    args : dict
        Dictionnary containing all parameters.

    Methods
    -------
    train_step:
        One training step for on batch of data.
    val:
        Validation for one epoch on the full validation dataset.
    training_loop:
        Full training loop for N epochs.
    """

    def __init__(self, args):
        self.loss = nn.CrossEntropyLoss()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = Classifier(args)
        self.nb_gpus = torch.cuda.device_count()
        # Handling multiple GPUs training:
        if self.nb_gpus > 1:
            print("Let's use", self.nb_gpus, "GPUs!")
            self.model = nn.DataParallel(self.model)
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, patience=5
        )
        tokenizer_path = args.tokenizer or args.model_unsupervised
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path, trust_remote_code=True, revision="main"
        )
        count_parameters(self.model)

    def train_step(self, input_ids, attention_mask, labels):
        """
        One training step for one batch of data.
        """
        output = self.model(input_ids, attention_mask)
        loss = self.loss(output, labels)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        preds = output.argmax(dim=1)
        return loss.item(), preds

    def resume(self, train_loader, val_loader, args):
        """Resume training using mentionned checkpoint"""
        # Load model dict :
        logger.info("Resuming training %s",args.resume)
        state_dict = torch.load(args.resume)
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            new_key = k.replace("module.", "")
            new_state_dict[new_key] = v
        self.model.load_state_dict(new_state_dict, strict=False)
        # Find last epoch :
        checkpoint_parts = os.path.basename(args.resume).split("_")
        if (
            len(checkpoint_parts) < 3
            or checkpoint_parts[0] != "model"
            or checkpoint_parts[1] != "epoch"
            or not checkpoint_parts[2].isdigit()
        ):
            raise ValueError(
                "Resume checkpoint must be named "
                "'model_epoch_<N>_<dataset>_pretrained_<bool>.pt'."
            )
        last_epoch = int(checkpoint_parts[2])
        logger.info("Last epoch of training was epoch : %s", last_epoch)
        args.start_epoch = last_epoch + 1
        args.output_dir = os.path.dirname(args.resume)
        logger.info("Resuming training from epoch %s", args.start_epoch)
        logger.info("Saving in %s", args.output_dir)
        # Launch training loop starting to the last epoch:
        self.training_loop(train_loader, val_loader, args)

    def val(self, val_loader, args):
        """
        Validation for one epoch on the full validation dataset.
        """
        self.model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                seq, labels = batch["sequence"], batch["label"]
                # Tokenization:
                tokens = self.tokenizer(
                    list(seq),
                    padding="max_length",
                    truncation=True,
                    max_length=args.max_length,
                    return_tensors="pt",
                )
                input_ids = tokens["input_ids"].to(self.device)
                attention_mask = tokens["attention_mask"].to(self.device)
                labels = labels.to(self.device)
                # Predictions:
                output = self.model(input_ids, attention_mask)
                loss = self.loss(output, labels)
                val_loss += loss.item()
                preds = output.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        val_acc = (correct / total) * 100
        return val_loss, val_acc

    def training_loop(self, train_loader, val_loader, args):
        """Full training loop for N epochs."""
        # Save used hyperparameters :
        save_args_to_json(args)
        # Starting with a high best loss to have less
        best_epoch_loss = 1000
        best_model_name = ""
        for epoch in range(args.start_epoch, args.epochs+1):
            self.model.train()
            # Init epoch metrics :
            epoch_loss = 0
            correct = 0
            total = 0
            for batch in tqdm(train_loader, desc=f"Epoch {epoch}"):
                seq, labels = batch["sequence"], batch["label"]
                tokens = self.tokenizer(
                    list(seq),
                    padding="max_length",
                    truncation=True,
                    max_length=args.max_length,
                    return_tensors="pt",
                )
                input_ids = tokens["input_ids"].to(self.device)
                attention_mask = tokens["attention_mask"].to(self.device)
                labels = labels.to(self.device)
                # Predictions:
                loss, preds = self.train_step(input_ids, attention_mask, labels)
                epoch_loss += loss
                correct += (preds == labels).sum().item()
                total += labels.size(0)
            # Compute epoch metrics:
            accuracy = (correct / total) * 100
            mean_epoch_loss = epoch_loss / len(train_loader)
            self.scheduler.step(mean_epoch_loss)
            # Validation:
            val_loss, val_acc = self.val(val_loader, args)
            logger.info(
                f"Epoch {epoch} | train Loss: {mean_epoch_loss:.4f} | "
                f"train Acc: {accuracy:.4f}% | "
                f"val Loss: {val_loss / len(val_loader):.4f} | val Acc: {val_acc:.4f}%"
                )
            if mean_epoch_loss < best_epoch_loss:
                if self.nb_gpus == 1 or self.nb_gpus == 0:
                    # Save the model :
                    model_name = get_save_model_path(args, epoch)
                    torch.save(self.model.state_dict(), model_name)
                    logger.info("Saved model to %s",model_name)
                    # Save only the classifier :
                    torch.save(
                        self.model.classifier.state_dict(),
                        get_save_classifier_path(args, epoch),
                    )
                    # Save only the dnabert2 for embeddings :
                    torch.save(
                        self.model.dnabert_2.state_dict(),
                        get_save_dnabert2_path(args, epoch),
                    )
                    best_epoch_loss = mean_epoch_loss
                    best_model_name = model_name
                else:
                    # Handling multiple GPUS with dataparallel
                    # Save the model :
                    model_name = get_save_model_path(args, epoch)
                    torch.save(self.model.module.state_dict(), model_name)
                    logger.info("Saved model to %s",model_name)
                    # Save only the classifier :
                    torch.save(
                        self.model.module.classifier.state_dict(),
                        get_save_classifier_path(args, epoch),
                    )
                    # Save only the dnaberts for embeddings :
                    torch.save(
                        self.model.module.dnabert_2.state_dict(),
                        get_save_dnabert2_path(args, epoch),
                    )
                    best_epoch_loss = mean_epoch_loss
                    best_model_name = model_name
        logger.info("Best model is %s", best_model_name)
