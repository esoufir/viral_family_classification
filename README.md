# Family-level classification of viral contigs using deep learning

## Description

This repository contains the finetuned version of DNABERT-2 (https://github.com/MAGICS-LAB/DNABERT_2) for viral taxonomy classification. 

This repository contains the training code and the model's weights of the publication []. 

## Getting Started

### Setting up

Models were trained using python and pytorch library. 

* Clone this project :


```
git clone https://github.com/esoufir/viral_family_classification.git
cd viral_family_classification
```

* The `environnement.yaml` file contains the full list of used librairies and packages. Use the it to install dependencies :

```
conda env create --file=environment.yaml
conda activate viral_family_classification
```

### Data formats

Run the commands below from the `src/` directory. Training data paths and model paths are interpreted relative to the current working directory.

Training and validation CSV files must contain these columns:

| Column | Description |
| --- | --- |
| `Contig` | Identifier for each contig |
| `Seq` | DNA sequence |
| `id_Family` | Numeric family label |

For prediction, a labeled CSV must also contain a `Family` column with the family name for each row. The `--family2id` file must be a CSV with `Family` and `id_Family` columns to map family names to class IDs. Alternatively, `--data_path` can point to a FASTA file: its record IDs and sequences are read directly, without labels.

### Predict with the model

The code for predicting is contained in the [`src/predict.py`](https://github.com/esoufir/viral_family_classification/tree/main/src/predict.py) file.  
```
cd src/
```

| Parameter | Required | Description |
| --- | --- | --- |
| `--data_path` | Yes | Input CSV or FASTA file. CSV input must include the columns described above. |
| `--model_path` | Yes | Path to the trained classifier checkpoint (`.pt`). |
| `--model_unsupervised` | Yes | Path to the pretrained DNABERT-2 model and tokenizer files. |
| `--batch_size` | Yes | Number of sequences processed per batch. |
| `--family2id` | Yes | CSV mapping families to IDs, with `Family` and `id_Family` columns. |
| `--conformal THRESHOLD` | No | Threshold for conformal prediction (for example, `0.9899547258391976`). Omit it to output top-1 predictions. |

The pretrained DNABERT-2 model is included in `models/DNABERT_2_pretrained/`. From `src/`, pass its local path using `--model_unsupervised`. For example, to evaluate predictions on a labeled CSV:

```
python predict.py --data_path ../toy_example/test.csv  --model_path ../models/model_train_split_on_sequences_DA.pt  --model_unsupervised ../models/DNABERT_2_pretrained/ --batch_size 10 --family2id ../toy_example/family2id.csv
```

To predict sequences from a FASTA file without labels, use the same local model path:

```
python predict.py --data_path ../toy_example/test.fasta --model_path ../models/model_train_split_on_sequences_DA.pt --model_unsupervised ../models/DNABERT_2_pretrained/ --batch_size 10 --family2id ../toy_example/family2id.csv
```



##### Conformal predictions

Especially, to perform conformal predictions, use the `--conformal` argument with the threshold used in the publication = 0.9899547258391976.

```
python predict.py --data_path ../toy_example/test.csv  --model_path ../models/model_train_split_on_sequences_DA.pt  --model_unsupervised ../models/DNABERT_2_pretrained/ --batch_size 10 --family2id ../toy_example/family2id.csv --conformal 0.9899547258391976
```


All prediction files are written to `src/predictions/`.


### Training code

The main training code is contained in the [`src/classification_dnabert2.py`](https://github.com/esoufir/viral_family_classification/tree/main/src/classification_dnabert2.py) file. 

| Parameter | Description |
| --- | --- |
| `--datapath` | Directory containing the training and validation CSV files. |
| `--train_dataname` | Training CSV filename. |
| `--val_dataname` | Validation CSV filename. |
| `--epochs` | Final epoch number for this run. |
| `--train_batch_size` | Training batch size. |
| `--val_batch_size` | Validation batch size. |
| `--start_epoch` | First epoch to run (set automatically when resuming). |
| `--lr` | Learning rate. |
| `--max_length` | Maximum token sequence length. |
| `--model_unsupervised` | Pretrained model path or Hugging Face identifier. For the included local weights, use `../models/DNABERT_2_pretrained/` from `src/`. |
| `--tokenizer` | Tokenizer path, if it is stored separately from the pretrained model. |
| `--label_column` | Name of the class-label column in the CSV files. |
| `--subsample_rank` | Optional CSV column used to select a rank when subsampling. |
| `--subsample` | Balance the sampled training examples by class. |
| `--freeze` | Freeze pretrained DNABERT-2 weights during training. |
| `--resume`| Path to a full-model checkpoint from which to continue training. |
| `--output_dir` | The script sets this to `outputs/<timestamp>/checkpoints`; a value passed on the command line is replaced. |

You can adjust the different hyperparameters to your liking and to suit your computationnal ressources. The training was made on GPU V100/A100.

For example, to train on the `toy_dataset` :
```
python classification_dnabert2.py --datapath ../toy_example/ \  
--train_dataname train_augmented.csv --val_dataname val.csv  \ --model_unsupervised ../models/DNABERT_2_pretrained/ \ --train_batch_size 10 --val_batch_size 10 --epochs 5
```


The tokenizer uses `--model_unsupervised` by default. Set `--tokenizer` only if the tokenizer files are in a different directory.

Each training run creates a timestamped directory under `src/outputs/`:

```
outputs/YYYY-MM-DD_HH-MM-SS/
├── checkpoints/
│   ├── model_epoch_<N>_<dataset>_pretrained_<bool>.pt
│   ├── model_epoch_<N>_<dataset>_pretrained_<bool>_classifier.pt
│   └── model_epoch_<N>_<dataset>_pretrained_<bool>_dnabert2.pt
└── logs/
    ├── classification_dnabert2.log
    └── arguments.json
```

The full-model checkpoint can be used with `predict.py` through `--model_path`. To continue training from one, pass its path to `--resume` and provide the training and model arguments again. `--epochs` is the final epoch number, not the number of additional epochs. For example, to continue from epoch 5 through epoch 10:


Resuming restores the model weights, but not the optimizer or scheduler state. New logs and saved arguments go in a new timestamped run directory. Resumed checkpoints are written to the directory containing the checkpoint passed to `--resume`.

## Acknowledgments

This project was provided with computer and storage resources by GENCI at IDRIS, thanks to the grant 2024/2025/2026-AD010315986 on the supercomputer Jean Zay's V100/A100 partition.

The authors acknowledge the National Network of Computing Resources of the Institut Français de Bioinformatique (IFB), funded by the Programme d’Investissements d’Avenir (PIA), grant Agence Nationale de la Recherche number ANR-11-INBS-0013, for providing HPC resources that have contributed to the research results reported within the paper.
