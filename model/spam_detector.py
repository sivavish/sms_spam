import torch
import torch.nn as nn


class SpamDetector(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim=128,
        hidden_dim=128,
        num_layers=1,
        dropout=0.3,
        bidirectional=True
    ):
        super(SpamDetector, self).__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=0
        )

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional
        )

        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim

        self.dropout = nn.Dropout(dropout)

        self.fc = nn.Linear(lstm_output_dim, 1)

    def forward(self, x):

        embedded = self.embedding(x)

        lstm_out, (hidden, cell) = self.lstm(embedded)

        if self.lstm.bidirectional:
            forward_hidden = hidden[-2]
            backward_hidden = hidden[-1]
            final_hidden = torch.cat((forward_hidden, backward_hidden), dim=1)
        else:
            final_hidden = hidden[-1]

        output = self.dropout(final_hidden)

        logits = self.fc(output)

        return logits.squeeze(1)
