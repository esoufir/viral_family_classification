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

### Predict with the model

The code is contained in the `src/` folder. 
```
cd src/
```

```
usage: predict.py [-h] [--data_path DATA_PATH] [--model_path MODEL_PATH]
                  [--model_unsupervised MODEL_UNSUPERVISED]
                  [--batch_size BATCH_SIZE] [--conformal [CONFORMAL]]
                  [--family2id FAMILY2ID] [--num_classes NUM_CLASSES]

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
  --num_classes NUM_CLASSES
                        Number of classes.```


To perform top-1 predictions on a viral contig FASTA file, run :


```

##### Conformal predictions

Especially, to perform conformal predictions, use the `--conformal` argument with the threshold

### Training code


```
python classification_dnabert2.py \
--datapath data/to_example \
--train_dataname train_augmented.csv \
--val_dataname val.csv \
--model_unsupervised DNABERT_2_pretrained  \
--train_batch_size 100 --val_batch_size 100 \
--epochs 30
```

You can adjust the different hyperparameters to your liking and to suit your computationnal ressources. 

For each launched training, a specific folder with the time code will be constructed in the òutputs`folder with the following folders :

```
outputs/YYYY-MM-DD_HH-MM-SS
├── checkpoints/
│   ├── model_epoch_N_checkpoints.pt
│   ├── model_epoch_N_checkpoints_dnabert2.pt
│   └── model_epoch_N_checkpoints_classifier.pt
├── logs/
│   ├── classification_dnabert2.log
│   └── arguments.json
└──predictions/
    └── predictions.csv
```


## License

This project is licensed under the [NAME HERE] License - see the LICENSE.md file for details

## Acknowledgments

