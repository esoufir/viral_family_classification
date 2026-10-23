from torch.utils.data import Dataset
import pandas as pd 
from torch.utils.data import DataLoader
import torch
from transformers import AutoModel, AutoTokenizer, BertModel
import csv
import numpy as np
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# TODO : Obsolet, see models.py and classification_dnabert2.py

# TODO : Implement label personalize using suffix for getting the taxonomic rank
# dnabert2_path = "/lustre/fswork/projects/rech/uos/unb69nk/dl_meta_vir/models/DNABERT_2/DNABERT_2_pretrained/"

# tokenizer = AutoTokenizer.from_pretrained(dnabert2_path, trust_remote_code=True)        
# dnabert_2 = BertModel.from_pretrained(dnabert2_path, trust_remote_code=True) 
device =  torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# def get_embedding(sequence):
#     inputs = tokenizer(sequence, return_tensors = 'pt',  padding="max_length", truncation=True, max_length=512)["input_ids"]
#     hidden_states = dnabert_2(inputs)[0] # [1, sequence_length, 768]
#     # embedding with mean pooling
#     embedding_mean = torch.mean(hidden_states[0], dim=0)
#     return embedding_mean

# class ViralSequenceDataset(Dataset):

    # def __init__(self, data_seq_path):
    #     self.df_sequences = pd.read_csv(data_seq_path)
    #     # Map labels to int for tensors afterwards
    #     self.label_to_idx = {label: idx for idx, label in enumerate(sorted(self.df_sequences["Label"].unique()))}

    # def __len__(self):
    #     return len(self.df_sequences)
        
    # def __getitem__(self, idx):
    #     # TODO : Implementation from reading directly from a FASTA ?, see where to put the labels in the header
    #     sequence = self.df_sequences["Sequence"].iloc[idx]
    #     label = self.df_sequences["Label"].iloc[idx] # generalized so we can do species, family, ... clustering
    #     # TODO : add here some potentential data augmentation at a random rate

    #     # Encode label to int
    #     label_idx = self.label_to_idx[label]

    #     # Convertit en tenseur
    #     label_tensor = torch.tensor(label_idx, dtype=torch.long)

    #     # Get the corresponding embedding for the sequence : 
    #     with torch.no_grad():
    #         sequence_embedding = get_embedding(sequence)
    #     return sequence_embedding, label_tensor # TODO : voir si c'est pas trop inutile de se renvoyer les tenseur ssur la meme device



class ViralSequenceSupervisedDataset(Dataset): 

    # Virus GENBANK accession -> 0
    # Contig-> 1
    # Seq -> 2
    # Realm -> 3
    # Kingdom -> 4 ,
    # Phylum -> 5
    # Class -> 6,
    # Order -> 7,
    # Family -> 8
    # Genus -> 9,
    # Species -> 10
    # Genome -> 11

    def __init__(self, data_path, tokenizer, sequence_column, label_column):
            # load data from the disk
            dataframe = pd.read_csv(data_path)
            dataframe = dataframe[dataframe[label_column] != -1] # TODO : l'anticiper des la création de la colonne 
            self.dataframe = dataframe
            # data is in the format of [AccessionID,Sequence,Label,family]
            print("Perform single sequence classification...")
            self.tokenizer = tokenizer
            self.sequences = self.dataframe[sequence_column].tolist()
            self.labels = self.dataframe[label_column].astype(int).tolist()
            self.num_labels = len(np.unique(self.labels))
            # assert torch.all(labels >= 0), f"Invalid label < 0 found: {self.labels}"
            # assert torch.all(labels < self.num_labels), f"Invalid label ≥ n_classes found: {labels}"
            print(f'{self.num_labels} labels found.')
            # print(np.unique(self.labels))

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        labels = self.labels[idx]
        labels = torch.tensor(labels, dtype=torch.long)

        
        # Tokenize a la volée for memory issues : 
        output = self.tokenizer(
                    self.sequences[idx],
                    return_tensors="pt",
                    padding="longest",
                    max_length=self.tokenizer.model_max_length,
                    truncation=True,
                )
        input_ids = output["input_ids"]
        attention_mask = output["attention_mask"]
        return dict(input_ids=input_ids.squeeze(0), attention_mask=attention_mask.squeeze(0), labels=labels)


class ViralEmbeddingDataset(Dataset):
    """Embeddings of DNABERT2 pretrained are obtained. """

    def __init__(self, data_emb_path):
        self.df_embeddings = pd.read_csv(data_emb_path)
        # Map labels to int for tensors afterwards
        self.label_to_idx = {label: idx for idx, label in enumerate(sorted(self.df_embeddings["Label"].unique()))}
        self.embedding_columns = [col for col in self.df_embeddings.columns if col.startswith("emb_")]

    def __len__(self):
        return len(self.df_embeddings)
    
    def __getitem__(self, idx):
        # Récupère directement les embeddings
        embedding_values = self.df_embeddings.loc[idx, self.embedding_columns].values.astype(float)  # numpy array
        embedding_tensor = torch.tensor(embedding_values, dtype=torch.float)

        label = self.df_embeddings.loc[idx, "Label"]
        label_idx = self.label_to_idx[label]
        label_tensor = torch.tensor(label_idx, dtype=torch.long)

        return embedding_tensor, label_tensor

def generate_data_loaders(training_dataset, validation_dataset, config):
    train_dataloader = DataLoader(training_dataset, batch_size=config["train_batch_size"], shuffle=True, num_workers=1)
    validation_dataloader = DataLoader(validation_dataset, batch_size=config["val_batch_size"], shuffle=True, num_workers=1)
    return train_dataloader, validation_dataloader


class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""


    def __init__(self, tokenizer):
        self.tokenizer=tokenizer

    def __call__(self, instances):
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.Tensor(labels).long()
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )