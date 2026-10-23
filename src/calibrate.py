import torch
import json
import sys
import logging
from tqdm import tqdm
from dataloader import contig_loader_csv_test
from transformers import AutoTokenizer
import pandas as pd
import numpy as np
from collections import defaultdict
from models import Classifier
from pathlib import Path
import rich_click as click
from types import SimpleNamespace
import torch.nn.functional as F


logger = logging.getLogger(__name__)


@click.command()
@click.option(
    "--model_path",
    type=click.Path(exists=True),
    required=True,
    help="Path to model weights (with classification model).",
)
@click.option(
    "--calibration_dataset",
    type=click.Path(exists=True),
    required=True,
    help="Path to contigs csv.",
)
@click.option(
    "--model_unsupervised",
    type=click.Path(exists=True),
    required=True,
    help="Path to the pretrained weights.",
)
@click.option(
    "--feat_dim",
    default=256,
    show_default=True,
    help="Dimension of one of the hidden layers of the classifier.",
)  # TODO : adapt this because the second layer is not parametrable
@click.option("--batch_size", default=256, show_default=True, help="Batch size")
@click.option(
    "--max_length", default=300, show_default=True, help="Longueur max des séquences"
)
@click.option(
    "--label_column", default="id_Family", show_default=True, help="Id label column"
)
@click.option(
    "--family2id",
    type=click.Path(exists=True),
    required=True,
    help="Path to family2id file csv.",
)
@click.option(
    "--cal2id",
    type=click.Path(exists=True),
    required=True,
    help="Path to family2id file csv.",
)
@click.option(
    "--other", is_flag=True, help="Wether its my dataset of test or not"
)  # TODO : plus généraliste parce que là c'est déguakasse
def main(
    model_path,
    calibration_dataset,
    model_unsupervised,
    feat_dim,
    batch_size,
    max_length,
    label_column,
    family2id,
    cal2id,
    other,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = Path(model_path)
    output_dir = model_path.parents[1].name
    print("Output dir", output_dir)

    # Preparing arguments for training
    args = SimpleNamespace(
        model_path=str(model_path),
        data_path=str(calibration_dataset),
        output_dir=output_dir,
        model_unsupervised=model_unsupervised,
        feat_dim=feat_dim,
        batch_size=batch_size,
        max_length=max_length,
        label_column=label_column,
        freeze=None,
        subsample_rank=None,  # subsample_rank
        num_classes=128,  # TODO : charger automatiquement en utilisant le repertoire du modele old =130
        other=other,
        family2id=pd.read_csv(family2id, index_col=0),
        cal2id=pd.read_csv(cal2id, index_col=0),
    )

    logging.basicConfig(
        filename=Path("outputs")
        / Path(output_dir)
        / "logs"
        / "conformal_prediction_dnabert2.log",
        level=logging.INFO,
        filemode="w",
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger.info("Calibrating DNABERT2 for conformal prediction")
    logger.info("Loading data...")
    test_loader = contig_loader_csv_test(args)
    logger.info(f"Dataname:{args.data_path}")

    # Load model and its dictionnary
    logger.info("Loading model...")
    logger.info(f"Model name:{args.output_dir}")
    model = Classifier(args)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    tokenizer = AutoTokenizer.from_pretrained(args.model_unsupervised)

    model.eval()
    model.to(device)
    ids, true_labels, good_pred_softmax = [], [], []
    i = 0
    with torch.no_grad():
        for batch in tqdm(
            test_loader, desc=f"Calibrating model for conformal prediction"
        ):
            id, seq, labels = batch["id"], batch["sequence"], batch["label"]
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
            # Get the softmax of correct predictions
            true_class_probs = probs.gather(1, labels.unsqueeze(1)).squeeze(1)
            # Adding to return stuf :
            ids.extend(id)
            good_pred_softmax.extend(true_class_probs.detach().cpu().tolist())
            true_labels.extend(labels.detach().cpu().tolist())

    df_predictions = {
        "Contig": ids,
        "softmax_score_correct_prediction": good_pred_softmax,
        "true_class": true_labels,
    }
    df_predictions = pd.DataFrame(
        df_predictions,
        columns=["Contig", "softmax_score_correct_prediction", "true_class"],
    )

    # Voir quelles true_class ne matchent pas
    missing = set(df_predictions["true_class"].unique()) - set(
        args.cal2id["id_Family"].values
    )
    print(f"true_class sans correspondance dans cal2id : {missing}")

    df_predictions = df_predictions.merge(
        args.cal2id, left_on="true_class", right_on="id_Family", how="left"
    )
    Path(f"outputs/{output_dir}/conformal").mkdir(parents=True, exist_ok=True)
    path_output_file = (
        f"outputs/{output_dir}/conformal/calibration_{Path(args.data_path).stem}.csv"
    )
    df_predictions.to_csv(path_output_file)
    logger.info(f"Calibration scores saved in {path_output_file}")
    # Classical conformal framework :
    # Calculate quantile
    n = len(df_predictions)
    alpha = 0.05
    q_level = np.ceil((n + 1) * (1 - alpha)) / n
    print("qlevel 2", q_level)
    q_level = min(q_level, 1.0)
    print("qlevel 2", q_level)
    # Calibration scores are softmax values of the correct predictions:
    cal_scores = 1 - np.array(good_pred_softmax)
    qhat = np.quantile(cal_scores, q_level, method="higher")
    conformal_info = []
    conformal_info.append({"true_class": None, "qhat": qhat})
    # Class wise conformal framework
    # for family in df_predictions["true_class"].unique():
    #     # Family size represents
    #     n_family = df_predictions[df_predictions["true_class"] ==  family].shape[0]
    #     q_level_family = np.ceil((n_family+1)*(1-alpha))/n_family
    #     print("nfam, qfma", n_family, q_level_family)
    #     cal_scores_family = df_predictions[df_predictions["true_class"] == family]["softmax_score_correct_prediction"]
    #     qhat_family = np.quantile(cal_scores_family, q_level_family)
    #     print(f"qhat family {family}", qhat_family)
    #     infos = {
    #         "true_class":family,
    #         "qhat":qhat_family,
    #     }
    #     conformal_info.append(infos)

    df_conformal = pd.DataFrame(conformal_info, columns=["true_class", "qhat"])
    print(df_conformal.head())
    df_conformal.to_csv(
        f"outputs/{output_dir}/conformal/qhat_{Path(args.data_path).stem}.csv"
    )
    # conformal_info = []
    # print(df_predictions["Order"].unique(), "---------------------")
    # for order in df_predictions["Order"].unique():
    #     # Filtrer les données de calibration pour cet ordre
    #     print("order :",order)
    #     cal_scores_order = df_predictions[df_predictions["Order"] == order]["softmax_score_correct_prediction"]
    #     n_order = len(cal_scores_order)
    #     q_level_order = np.ceil((n_order + 1) * (1 - alpha)) / n_order
    #     q_level_order = min(q_level_order, 1.0)  # clamp pour éviter erreur np.quantile

    #     qhat_order = np.quantile(cal_scores_order, q_level_order)
    #     print(f"Order={order} | n={n_order} | q_level={q_level_order:.3f} | qhat={qhat_order:.4f}")
    #     conformal_info.append({
    #         "Order": order,
    #         "qhat": qhat_order,
    #         "n_cal": n_order,
    #     })

    # df_conformal = pd.DataFrame(conformal_info, columns=["Order", "qhat", "n_cal"])

    # print(df_conformal)

    # Ca sera pour la prediction ca :
    #  val_smx = model(val_X).softmax(dim=1).numpy()
    # prediction_sets = val_smx >= (1-qhat) # 3: form prediction sets


if __name__ == "__main__":
    main()
