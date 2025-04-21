import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler

class StockLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout_rate=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_rate)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])  
        return self.fc(out)


def prepare_lstm_data(df, feature_cols, target_col, sequence_length=30):
    df = df.dropna(subset=feature_cols + [target_col])
    scaler = MinMaxScaler()
    features = scaler.fit_transform(df[feature_cols])
    targets = df[target_col].values

    X_seq, y_seq = [], []
    for i in range(len(df) - sequence_length):
        X_seq.append(features[i:i+sequence_length])
        y_seq.append(targets[i+sequence_length])

    return torch.tensor(X_seq, dtype=torch.float32), torch.tensor(y_seq, dtype=torch.float32).view(-1, 1)

def train_lstm_model(X, y, input_size, epochs=20, lr=0.001, dropout_rate=0.3):
    model = StockLSTM(input_size, dropout_rate=dropout_rate)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()
        if epoch % 5 == 0:
            print(f"Epoch {epoch} | Loss: {loss.item():.6f}")

    return model


def monte_carlo_lstm_predict(model, x_input, iterations=1000):
    model.train()  # Enable dropout even during inference
    preds = []

    for _ in range(iterations):
        with torch.no_grad():
            pred = model(x_input).item()
            preds.append(pred)

    return {
        "mean": np.mean(preds),
        "std": np.std(preds),
        "samples": preds
    }
