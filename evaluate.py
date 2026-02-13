import torch
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix

from utils.preprocessing import preprocess_dataframe
from model.data_loader import (
    stratified_split,
    build_vocab,
    SMSDataset
)
from model.spam_detector import SpamDetector


BATCH_SIZE = 32
MAX_LENGTH = 60
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")



df = pd.read_csv("data/spam.csv", encoding="latin-1")
df = preprocess_dataframe(df)

train_df, val_df, test_df = stratified_split(df)


vocab = build_vocab(train_df["message"])
vocab_size = len(vocab)


test_dataset = SMSDataset(
    test_df["message"],
    test_df["label"],
    vocab,
    MAX_LENGTH
)

test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)


model = SpamDetector(vocab_size=vocab_size)
model.load_state_dict(torch.load("best_model.pt", map_location=DEVICE))
model = model.to(DEVICE)
model.eval()



all_preds = []
all_labels = []

with torch.no_grad():
    for texts, labels in test_loader:
        texts = texts.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(texts)
        probs = torch.sigmoid(outputs)

        preds = (probs > 0.5).float()

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())


print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=["Ham", "Spam"]))

print("\nConfusion Matrix:")
print(confusion_matrix(all_labels, all_preds))
