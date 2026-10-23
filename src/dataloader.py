"""
Handling contig data for training.
"""

import os
import sys
import csv
from collections import Counter
import numpy as np
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from Bio import SeqIO

csv.field_size_limit(sys.maxsize)


class Samples(Dataset):
    def __init__(self, ids, sequences, labels, subsample=False, rank_subsample=None):
        self.ids = ids
        self.sequences = sequences
        self.labels = labels
        self.minority_classes = None
        if subsample:
            self.subsample = True
            self.rank_subsample = rank_subsample
        else:
            self.subsample = False

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        """Mapping with seq and label for performing classification."""
        seq_id = self.ids[idx]
        seq = self.sequences[idx]
        label = int(float(self.labels[idx]))
        if self.subsample:
            return {
                "id": seq_id,
                "sequence": seq,
                "label": label,
                "subsample_rank": [self.rank_subsample[idx]],
            }
        return {"id": seq_id, "sequence": seq, "label": label}

    def find_minority_classes(self):
        label_counts = Counter(self.labels)
        counts = np.array(list(label_counts.values()))
        threshold = np.quantile(counts, 0.25)
        minority_classes = [
            label for label, count in label_counts.items() if count <= threshold
        ]
        return minority_classes


class SamplesNames(Dataset):
    def __init__(self, ids, sequences, labels, names, family2id):
        self.ids = ids
        self.sequences = sequences
        self.labels = labels
        self.names = names
        self.family2id_dict = dict(
            zip(family2id["Family"], family2id["id_Family"])
        )  # reindex for correct mapping
        self.unknown_id = max(self.family2id_dict.values()) + 1
        self.labels = [
            self.family2id_dict.get(name, self.unknown_id) for name in self.names
        ]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        """Mapping with seq and label for performing classification."""
        seq_id = self.ids[idx]
        seq = self.sequences[idx]
        label = int(float(self.labels[idx]))
        label_name = self.names[idx]
        return {"id": seq_id, "sequence": seq, "label": label, "label_name": label_name}

    def find_minority_classes(self):
        label_counts = Counter(self.labels)
        counts = np.array(list(label_counts.values()))
        threshold = np.quantile(counts, 0.25)
        minority_classes = [
            label for label, count in label_counts.items() if count <= threshold
        ]
        return minority_classes


class SamplesWithoutLabels(Dataset):
    def __init__(self, ids, sequences):
        """Sequence class for unlabeled data, like for predictions."""
        self.ids = ids
        self.sequences = sequences

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        """Mapping with seq and label for performing classification."""
        seq_id = self.ids[idx]
        seq = self.sequences[idx]
        return {"id": seq_id, "sequence": seq}


def contig_loader_csv(args, load_train=True):
    """No more pairs in the dataset"""
    delimiter = ","
    reader = None
    filepath = os.path.join(
        args.datapath, args.train_dataname if load_train else args.val_dataname
    )
    with open(filepath, encoding="utf-8", newline="") as csvfile:
        reader = csv.DictReader(csvfile, delimiter=delimiter)
        data = list(reader)
    args.sequence_column = "Seq"
    args.ids = "Contig"
    sequences = [row[args.sequence_column] for row in data]
    labels = [row[args.label_column] for row in data]
    ids = [row[args.ids] for row in data]
    rank_subsample = None
    subsample = False

    if args.subsample_rank:
        rank_subsample = [row[args.subsample_rank] for row in data]
        subsample = True
    dataset = Samples(
        ids, sequences, labels, subsample=subsample, rank_subsample=rank_subsample
    )
    sampler = None
    if args.subsample & load_train:
        labels_idx = np.array(dataset.labels, dtype=np.int64)
        # count effectives by classes
        class_counts = np.bincount(labels_idx)
        # Use its inverse so less frequent will have higher weights
        class_weights = 1.0 / class_counts
        samples_weight = class_weights[labels_idx]  # poids pour chaque exemple
        # replace = true  -> can sample multiple times the same sequence
        sampler = WeightedRandomSampler(
            weights=samples_weight,
            num_samples=len(
                samples_weight
            ),
            replacement=True,
        )

    if load_train:
        if args.subsample:
            loader = DataLoader(
                dataset,
                batch_size=args.train_batch_size,
                num_workers=8,
                #drop_last=True,
                sampler=sampler,
            )
        else:
            loader = DataLoader(
                dataset,
                batch_size=args.train_batch_size,
                shuffle=True,
                num_workers=8,
                #drop_last=True,
            )

    else:
        loader = DataLoader(
            dataset,
            batch_size=args.val_batch_size,
            shuffle=True,
            num_workers=8,
            # drop_last=True,
        )
    return loader


def contig_loader_csv_test(args):
    """No more pairs in the dataset"""
    delimiter = ","
    reader = None
    filepath = args.data_path
    with open(filepath, encoding="utf-8",newline="") as csvfile:
        reader = csv.DictReader(csvfile, delimiter=delimiter)
        data = list(reader)
    args.sequence_column = "Seq"
    args.ids = "Contig"
    sequences = [row[args.sequence_column] for row in data]
    labels = [row[args.label_column] for row in data]
    label_names = [
        row["Family"] for row in data
    ]
    ids = [row[args.ids] for row in data]
    dataset = SamplesNames(ids, sequences, labels, label_names, args.family2id)
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=8)
    return loader


def contig_loader_fasta(args):
    """Construct data loader from FASTA."""
    ids,sequences=[],[]
    with open(args.data_path) as handle:
        for record in SeqIO.parse(handle, "fasta"):
            ids.append(record.id)
            sequences.append(str(record.seq))
    args.sequence_column = "Seq"
    args.ids = "Contig"
    dataset = SamplesWithoutLabels(ids, sequences)
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=4)
    return loader


def contig_loader_csv_calibration(args):
    """No more pairs in the dataset"""
    delimiter = ","
    reader = None
    filepath = args.calibration_path
    with open(filepath,encoding="utf-8", newline="") as csvfile:
        reader = csv.DictReader(csvfile, delimiter=delimiter)
        data = list(reader)

    args.sequence_column = "Seq"
    args.ids = "Contig"
    sequences = [row[args.sequence_column] for row in data]
    labels = [row[args.label_column] for row in data]
    ids = [row[args.ids] for row in data]

    dataset = Samples(ids, sequences, labels)
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=8)
    return loader


def contig_loader_csv_test_no_labels(args):
    """No more pairs in the dataset"""
    delimiter = ","
    reader = None
    filepath = args.data_path
    with open(filepath, encoding="utf-8", newline="") as csvfile:
        reader = csv.DictReader(csvfile, delimiter=delimiter)
        data = list(reader)

    args.sequence_column = "Seq"
    args.ids = "Contig"
    sequences = [row[args.sequence_column] for row in data]
    ids = [row[args.ids] for row in data]
    dataset = SamplesWithoutLabels(ids, sequences)
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=8)
    return loader
