# Family-level classification of viral contigs using deep learning

## Description

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

For prediction, a labeled CSV must also contain a `Family` column with the family name for each row. The `--family2id` file must be a CSV with `Family` and `id_Family` columns to compute . Alternatively, `--data_path` can point to a FASTA file : its record IDs and sequences are read directly, without labels.

### Predict with the model

The code is contained in the `src/` folder. 
```
cd src/
```

```
usage: predict.py [-h] [--data_path DATA_PATH] [--model_path MODEL_PATH] [--model_unsupervised MODEL_UNSUPERVISED] [--batch_size BATCH_SIZE] [--conformal [CONFORMAL]] [--family2id FAMILY2ID]

optional arguments:
  -h, --help            show this help message and exit
  --data_path DATA_PATH
                        Path to data on which to predict (CSV OR FASTA).
  --model_path MODEL_PATH
                        Path to model's weights.
  --model_unsupervised MODEL_UNSUPERVISED
                        Path to model's weights pretrained for tokenizer.
  --batch_size BATCH_SIZE
                        Train batch size.
  --conformal [CONFORMAL]
                        Optional conformal prediction threshold.
  --family2id FAMILY2ID
                        Family 2 id correspondance file
```

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


```
usage: classification_dnabert2.py [-h] [--datapath DATAPATH] [--train_dataname TRAIN_DATANAME] [--val_dataname VAL_DATANAME] [--train_batch_size TRAIN_BATCH_SIZE]
                                  [--val_batch_size VAL_BATCH_SIZE] [--epochs EPOCHS] [--start_epoch START_EPOCH] [--lr LR] [--max_length MAX_LENGTH] [--output_dir OUTPUT_DIR]
                                  [--tokenizer TOKENIZER] [--model_unsupervised MODEL_UNSUPERVISED] [--label_column LABEL_COLUMN] [--subsample_rank SUBSAMPLE_RANK] [--freeze]
                                  [--subsample] [--resume RESUME]

optional arguments:
  -h, --help            show this help message and exit
  --datapath DATAPATH   Path to data folder.
  --train_dataname TRAIN_DATANAME
                        Training data filename.
  --val_dataname VAL_DATANAME
                        Validation data filename.
  --train_batch_size TRAIN_BATCH_SIZE
                        Train batch size.
  --val_batch_size VAL_BATCH_SIZE
                        Validation batch size.
  --epochs EPOCHS       Number of epochs for training.
  --start_epoch START_EPOCH
                        Init epoch to start the training.
  --lr LR               Learning rate value.
  --max_length MAX_LENGTH
                        Max length of tokens
  --output_dir OUTPUT_DIR
                        Directory to save model checkpoints
  --tokenizer TOKENIZER
                        Path to used tokenizer (defaults to --model_unsupervised)
  --model_unsupervised MODEL_UNSUPERVISED
                        Path to used pretrained weights.
  --label_column LABEL_COLUMN
                        Column name of the label in the csv file.
  --subsample_rank SUBSAMPLE_RANK
                        Subsample the training data to only include a specific rank
  --freeze              Freeze the pretrained weights
  --subsample           Subsample the training data
  --resume RESUME       Resume the training using the given checkpoint.
```

You can adjust the different hyperparameters to your liking and to suit your computationnal ressources. The training was made on GPU V100/A100.

For example, to train on the `toy_dataset` :
```
python classification_dnabert2.py --datapath ../toy_example/ --train_dataname train_augmented.csv --val_dataname val.csv --model_unsupervised ../models/DNABERT_2_pretrained/ --train_batch_size 10 --val_batch_size 10 --epochs 5
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

```
python classification_dnabert2.py --datapath ../toy_example/ --train_dataname train_augmented.csv --val_dataname val.csv --model_unsupervised ../models/DNABERT_2_pretrained/ --train_batch_size 10 --val_batch_size 10 --epochs 10 --resume ../outputs/YYYY-MM-DD_HH-MM-SS/checkpoints/model_epoch_5_train_augmented_pretrained_True.pt
```

Resuming restores the model weights, but not the optimizer or scheduler state. New logs and saved arguments go in a new timestamped run directory; resumed checkpoints are written to the directory containing the checkpoint passed to `--resume`.

## Acknowledgments

This project was provided with computer and storage resources by GENCI at IDRIS, thanks to the grant 2024/2025/2026-AD010315986 on the supercomputer Jean Zay's V100/A100 partition.

The authors acknowledge the National Network of Computing Resources of the Institut Français de Bioinformatique (IFB), funded by the Programme d’Investissements d’Avenir (PIA), grant Agence Nationale de la Recherche number ANR-11-INBS-0013, for providing HPC resources that have contributed to the research results reported within the paper.
