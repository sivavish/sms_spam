import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import f1_score

from utils.preprocessing import preprocess_dataframe
from utils.class_weights import compute_pos_weight
from model.data_loader import (
    stratified_split,
    build_vocab,
    SMSDataset
)
from model.spam_detector import SpamDetector



BATCH_SIZE = 32
EPOCHS = 15
LEARNING_RATE = 0.001
MAX_LENGTH = 60
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


df = pd.read_csv("data/spam.csv", encoding="latin-1")
df = preprocess_dataframe(df)

train_df, val_df, test_df = stratified_split(df)


vocab = build_vocab(train_df["message"])
vocab_size = len(vocab)


train_dataset = SMSDataset(
    train_df["message"],
    train_df["label"],
    vocab,
    MAX_LENGTH
)

val_dataset = SMSDataset(
    val_df["message"],
    val_df["label"],
    vocab,
    MAX_LENGTH
)

test_dataset = SMSDataset(
    test_df["message"],
    test_df["label"],
    vocab,
    MAX_LENGTH
)


train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)



model = SpamDetector(vocab_size=vocab_size)
model = model.to(DEVICE)


pos_weight = compute_pos_weight(train_df["label"])
pos_weight = pos_weight.to(DEVICE)

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)


best_val_f1 = 0
patience = 3
early_stop_counter = 0

for epoch in range(EPOCHS):

    model.train()
    train_loss = 0

    for texts, labels in tqdm(train_loader):
        texts = texts.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(texts)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    avg_train_loss = train_loss / len(train_loader)


    model.eval()
    val_preds = []
    val_labels = []

    with torch.no_grad():
        for texts, labels in val_loader:
            texts = texts.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(texts)
            probs = torch.sigmoid(outputs)

            preds = (probs > 0.5).float()

            val_preds.extend(preds.cpu().numpy())
            val_labels.extend(labels.cpu().numpy())

    val_f1 = f1_score(val_labels, val_preds)

    print(f"\nEpoch {epoch+1}")
    print(f"Train Loss: {avg_train_loss:.4f}")
    print(f"Validation F1 (Spam): {val_f1:.4f}")

    # Early stopping
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        early_stop_counter = 0
        torch.save(model.state_dict(), "best_model.pt")
    else:
        early_stop_counter += 1

    if early_stop_counter >= patience:
        print("Early stopping triggered.")
        break


print("Training complete.")
