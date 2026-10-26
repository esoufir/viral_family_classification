"""
Code for predicting viral family classification using a trained model.
"""

import logging
import sys
import csv
import argparse
import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from transformers import AutoTokenizer
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
    classification_report,
)
from pathlib import Path
from dataloader import contig_loader_csv_test, contig_loader_fasta, SamplesNames
from models import Classifier

logger = logging.getLogger(__name__)


def get_args():
    """Arguments parser for predictions."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", type=str, help="Path to data on which to predict (CSV OR FASTA).")
    parser.add_argument("--model_path", type=str, help="Path to model's weights.")
    parser.add_argument("--model_unsupervised", type=str, help="Path to model's weights pretrained for tokenizer.")
    parser.add_argument("--batch_size", type=int, help="Train batch size.")
    parser.add_argument(
        "--conformal",
        nargs="?",
        type=float,
        help="Optional conformal prediction threshold."
    )
    parser.add_argument("--family2id", type=str, help="Family 2 id correspondance file")
    args = parser.parse_args()
    if not vars(args) or all(v is None for v in vars(args).values()):
        parser.print_help()
        sys.exit(0)
    return args


def predict(model, test_loader, tokenizer, device, args, output_dir):
    """Predict top-1 class by batch using the model and its arguments."""
    with torch.no_grad():
        # When lables are availables :
        if isinstance(test_loader.dataset, SamplesNames):
            predictions, probas, true_labels = [], [], []
            for batch in tqdm(test_loader, desc="Predicting"):
                seq, labels = batch["sequence"], batch["label"]
                tokens = tokenizer(
                        list(seq),
                        padding="max_length",
                        truncation=True,
                        max_length=args.max_length,
                        return_tensors="pt",
                    )
                input_ids = tokens["input_ids"].to(device)
                attention_mask = tokens["attention_mask"].to(device)
                labels = labels.to(device)
                output = model(input_ids, attention_mask)
                probs = F.softmax(output, dim=1)
                preds = output.argmax(dim=1)
                proba_max = torch.max(probs, dim=1).values
                probas.extend(proba_max.cpu().numpy())
                predictions.extend(preds.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())
            # Compute evaluation metrics:
            logger.info("Accuracy : %s", accuracy_score(true_labels, predictions))
            logger.info(
                "Balanced Accuracy : %s", balanced_accuracy_score(true_labels, predictions)
            )
            logger.info(classification_report(true_labels, predictions, zero_division=0))
            precision, recall, f1, _ = precision_recall_fscore_support(
                true_labels, predictions, average="macro", zero_division=0
            )
            logger.info("Precision (macro) : %s", precision)
            logger.info("Recall (macro): %s", recall)
            logger.info("F1-score (macro)  : %s", f1)

            df_inputs = pd.read_csv(args.data_file)
            true_labels_reindexed = df_inputs[args.rank].map(args.family_to_id).tolist()
            predictions_family = [args.id_to_family.get(p) for p in predictions]

            df_predictions = {
                "Contig": df_inputs["Contig"],
                "predicted_id": predictions,
                "predicted_class_name": predictions_family,
                "proba_max": probas,
                "true_class": true_labels_reindexed,
                "true_class_name": df_inputs[f"{args.rank}"],
            }
            df_predictions = pd.DataFrame(df_predictions)
            path_output_file = f"{output_dir}/predictions_{Path(args.data_file).stem}.csv"
            df_predictions.to_csv(path_output_file, index=None)
            logger.info("Predictions saved in %s", path_output_file)
        # When labels are not availables:
        else: 
            ids, predictions, probas = [], [], []
            for batch in tqdm(test_loader, desc="Predicting"):
                id, seq = batch["id"], batch["sequence"]
                tokens = tokenizer(
                    list(seq),
                    padding="max_length",
                    truncation=True,
                    max_length=args.max_length,
                    return_tensors="pt",
                )
                input_ids = tokens["input_ids"].to(device)
                attention_mask = tokens["attention_mask"].to(device)
                output = model(input_ids, attention_mask)
                probs = F.softmax(output, dim=1)
                preds = output.argmax(dim=1)
                proba_max = torch.max(probs, dim=1).values
                ids.extend(id)
                probas.extend(proba_max.cpu().numpy())
                predictions.extend(preds.cpu().numpy())
            predictions_family = [args.id_to_family.get(p) for p in predictions]
            df_predictions = {
                "Contig": ids,
                "predicted_id": predictions,
                "predicted_class_name": predictions_family,
                "proba_max": probas,
            }
            df_predictions = pd.DataFrame(df_predictions)
            path_output_file = f"{output_dir}/predictions_{Path(args.data_file).stem}.csv"
            df_predictions.to_csv(path_output_file, index=None)
            logger.info("Predictions saved in %s", path_output_file)

    return df_predictions


def predict_conformal(model, test_loader, tokenizer, device, args, output_dir):
    """Predict conformal predictions by batch using the model and its arguments."""
    path_output_file = f"{output_dir}/conformal_predictions_{Path(args.data_file).stem}_calibration_{str(args.conformal)}.csv"

    with torch.no_grad(), open(path_output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if isinstance(test_loader.dataset, SamplesNames):
            # header
            header = [
                "Contig",
                "label_name_true",
                "true_label",
                "prediction",
                "label_prediction",
                "proba_max",
                "conformal_pred_ids",
                "conformal_pred_classes",
            ]
            writer.writerow(header)
            for batch in tqdm(test_loader, desc="Predicting"):
                ids, seq, labels, label_names = (
                    batch["id"],
                    batch["sequence"],
                    batch["label"],
                    batch["label_name"],
                )
                tokens = tokenizer(
                    list(seq),
                    padding="max_length",
                    truncation=True,
                    max_length=args.max_length,
                    return_tensors="pt",
                )
                input_ids = tokens["input_ids"].to(device)
                attention_mask = tokens["attention_mask"].to(device)
                labels = labels.to(device)
                # To get one prediction i.e the highest softmax:
                output = model(input_ids, attention_mask)
                probs = F.softmax(output, dim=1)
                preds = output.argmax(dim=1)
                proba_max = torch.max(probs, dim=1).values.cpu().numpy()

                for i in range(len(ids)):
                    prob_i = probs[i]
                    pred_i = preds[i].item()
                    label_i = labels[i].item()
                    # Get conformal predictions
                    mask = prob_i >= (1 - args.conformal)
                    indices = torch.where(mask)[0].cpu().tolist()
                    labels_pred = [args.family2id.loc[idx, "Family"] for idx in indices]
                    # Sorting by decreasing softmax scores :
                    prob_i_to_sort = prob_i[indices].cpu().numpy()
                    idx_sort = np.argsort(prob_i_to_sort)[::-1]
                    sorted_labels = np.array(labels_pred)[idx_sort]
                    writer.writerow(
                        [
                            ids[i],  # ID contig
                            label_names[i],  # True label str
                            label_i,  # true label int
                            pred_i,  # predicted label int
                            args.family2id.loc[pred_i, "Family"],  # predicted label str
                            proba_max[i].item(),  # softmax max float
                            "|".join(map(str, prob_i_to_sort[idx_sort])),
                            "|".join(sorted_labels),
                        ]
                    )
                    f.flush()
        else : 
            # header
            header = [ "Contig", "prediction", "label_prediction",
                "proba_max","conformal_pred_ids","conformal_pred_classes"]
            writer.writerow(header)
            for batch in tqdm(test_loader, desc="Predicting"):
                ids, seq = batch["id"],batch["sequence"]
                tokens = tokenizer(
                    list(seq),
                    padding="max_length",
                    truncation=True,
                    max_length=args.max_length,
                    return_tensors="pt",
                )
                input_ids = tokens["input_ids"].to(device)
                attention_mask = tokens["attention_mask"].to(device)
                # To get one prediction i.e the highest softmax:
                output = model(input_ids, attention_mask)
                probs = F.softmax(output, dim=1)
                preds = output.argmax(dim=1)
                proba_max = torch.max(probs, dim=1).values.cpu().numpy()

                for i in range(len(ids)):
                    prob_i = probs[i]
                    pred_i = preds[i].item()
                    # Get conformal predictions
                    mask = prob_i >= (1 - args.conformal)
                    indices = torch.where(mask)[0].cpu().tolist()
                    labels_pred = [args.family2id.loc[idx, "Family"] for idx in indices]
                    # Sorting by decreasing softmax scores :
                    prob_i_to_sort = prob_i[indices].cpu().numpy()
                    idx_sort = np.argsort(prob_i_to_sort)[::-1]
                    sorted_labels = np.array(labels_pred)[idx_sort]
                    writer.writerow(
                        [
                            ids[i],  # ID contig
                            pred_i,  # predicted label int
                            args.family2id.loc[pred_i, "Family"],  # predicted label str
                            proba_max[i].item(),  # softmax max float
                            "|".join(map(str, prob_i_to_sort[idx_sort])),
                            "|".join(sorted_labels),
                        ]
                    )
                    f.flush()


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = Path(args.model_path)
    output_dir = Path("predictions")
    df_labels = pd.read_csv(args.family2id)
    df_labels["id_Family"] = df_labels["id_Family"].apply(lambda x: int(x))
    family_to_id = df_labels.set_index("Family")["id_Family"].to_dict()
    id_to_family = df_labels.set_index("id_Family")["Family"].to_dict()
    args.freeze = None
    args.family2id=df_labels
    args.family_to_id=family_to_id
    args.id_to_family=id_to_family
    args.label_column="id_Family"
    args.max_length = 300
    args.is_fasta = False
    args.rank="Family"
    args.num_classes = len(family_to_id)

    log_dir = Path("predictions")
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=log_dir / "predict_dnabert2.log",
        level=logging.INFO,
        filemode="w",
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    
    logger.info("Classification prediction")
    logger.info("Loading data...")
    if args.data_file.endswith(".csv"):
        test_loader = contig_loader_csv_test(args)
    else:
        test_loader = contig_loader_fasta(args)
        args.is_fasta=True
    logger.info("Dataname:%s", args.data_file)
    # Load model and its dictionnary
    logger.info("Loading model...")
    logger.info("Model name:%s", output_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Classifier(args)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    tokenizer = AutoTokenizer.from_pretrained(args.model_unsupervised)
    model.eval()
    model.to(device)
    # Perform asked prediction type:
    if args.conformal:
        logger.info("Conformal prediction threshold :%s", str(args.conformal))
        predict_conformal(
            model, test_loader, tokenizer, device, args, output_dir
        )
    else:
        predict(model, test_loader, tokenizer, device, args, output_dir)


if __name__ == "__main__":
    args = get_args()
    main(args)
