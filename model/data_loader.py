import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from collections import Counter


def stratified_split(df, test_size=0.15, val_size=0.15, random_state=42):

    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df["label"],
        random_state=random_state
    )

    train_df, val_df = train_test_split(
        train_df,
        test_size=val_size / (1 - test_size),
        stratify=train_df["label"],
        random_state=random_state
    )

    return train_df, val_df, test_df


def build_vocab(texts, min_freq=1):
    counter = Counter()

    for text in texts:
        tokens = text.split()
        counter.update(tokens)

    vocab = {"<PAD>": 0, "<UNK>": 1}
    idx = 2

    for word, freq in counter.items():
        if freq >= min_freq:
            vocab[word] = idx
            idx += 1

    return vocab


def text_to_sequence(text, vocab):
    tokens = text.split()
    sequence = []

    for token in tokens:
        if token in vocab:
            sequence.append(vocab[token])
        else:
            sequence.append(vocab["<UNK>"])

    return sequence


def pad_sequence(sequence, max_length):
    if len(sequence) < max_length:
        sequence = sequence + [0] * (max_length - len(sequence))
    else:
        sequence = sequence[:max_length]

    return sequence


class SMSDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_length):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts.iloc[idx]
        label = self.labels.iloc[idx]

        sequence = text_to_sequence(text, self.vocab)
        padded_sequence = pad_sequence(sequence, self.max_length)

        return (
            torch.tensor(padded_sequence, dtype=torch.long),
            torch.tensor(label, dtype=torch.float)
        )
