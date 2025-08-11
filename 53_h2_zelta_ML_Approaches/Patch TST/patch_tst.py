

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

# Define metrics
def calculate_metrics(y_true, y_pred):
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    mpe = np.mean((y_true - y_pred) / y_true) * 100
    return mse, rmse, mape, mpe

# Load and preprocess BTC data
data = pd.read_csv('/kaggle/input/adsgdhf/ethusdt_1h.csv')
data = data[['close']]

# Calculate SMA and EMA with window size 10
window_size = 10
data['sma'] = data['close'].rolling(window=window_size).mean()
data['ema'] = data['close'].ewm(span=window_size, adjust=False).mean()

# Drop NaN values caused by SMA and EMA calculations
data.dropna(inplace=True)

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data[['close', 'sma', 'ema']])

# Create sequences for time series prediction
def create_sequences(data, seq_length, target_column):
    sequences = []
    targets = []
    for i in range(len(data) - seq_length):
        seq = data[i:i + seq_length, 0]  # Input: close prices
        target = data[i + seq_length, target_column]  # Target: close, SMA, or EMA
        sequences.append(seq)
        targets.append(target)
    return np.array(sequences), np.array(targets)

seq_length = 1024

# Create datasets for close, SMA, and EMA
X_close, y_close = create_sequences(data_scaled, seq_length, target_column=0)
X_sma, y_sma = create_sequences(data_scaled, seq_length, target_column=1)
X_ema, y_ema = create_sequences(data_scaled, seq_length, target_column=2)

# Split data into train and test sets
def split_data(X, y):
    return train_test_split(X, y, test_size=0.2, random_state=42)

X_train_close, X_test_close, y_train_close, y_test_close = split_data(X_close, y_close)
X_train_sma, X_test_sma, y_train_sma, y_test_sma = split_data(X_sma, y_sma)
X_train_ema, X_test_ema, y_train_ema, y_test_ema = split_data(X_ema, y_ema)

# Convert data to PyTorch tensors
def to_tensor(X, y):
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

X_train_tensor_close, y_train_tensor_close = to_tensor(X_train_close, y_train_close)
X_test_tensor_close, y_test_tensor_close = to_tensor(X_test_close, y_test_close)
X_train_tensor_sma, y_train_tensor_sma = to_tensor(X_train_sma, y_train_sma)
X_test_tensor_sma, y_test_tensor_sma = to_tensor(X_test_sma, y_test_sma)
X_train_tensor_ema, y_train_tensor_ema = to_tensor(X_train_ema, y_train_ema)
X_test_tensor_ema, y_test_tensor_ema = to_tensor(X_test_ema, y_test_ema)

# Define PatchTST Model
class PatchTST(nn.Module):
    def __init__(self, input_dim, seq_len, patch_len, embed_dim, n_heads, n_layers, dropout=0.1):
        super(PatchTST, self).__init__()
        self.patch_len = patch_len
        self.n_patches = seq_len // patch_len

        # Linear embedding for patches
        self.embedding = nn.Linear(patch_len, embed_dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=n_heads,
                dim_feedforward=512,
                activation='gelu',
                dropout=dropout,
            ),
            num_layers=n_layers
        )
        self.fc = nn.Linear(embed_dim * self.n_patches, 1)  # Predict the next value

    def forward(self, x):
        batch_size = x.shape[0]
        # Split input into patches
        patches = x.view(batch_size, self.n_patches, self.patch_len)
        embeddings = self.embedding(patches)
        embeddings = embeddings.permute(1, 0, 2)  # (seq_len, batch, embed_dim)

        # Transformer
        transformer_out = self.transformer(embeddings)
        transformer_out = transformer_out.permute(1, 0, 2).reshape(batch_size, -1)

        # Final fully connected layer
        output = self.fc(transformer_out)
        return output

# Hyperparameters
input_dim = 1
seq_len = seq_length
patch_len = 1024
embed_dim = 512
n_heads = 8
n_layers = 6
learning_rate = 0.0001
num_epochs = 20
dropout = 0.2
# learning_rate = 0.0001
# num_epochs = 100
accumulation_steps =128
max_grad_norm = 1.0

# Train and evaluate with gradient accumulation, clipping, and regularization
def train_and_evaluate(X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor):
    model = PatchTST(input_dim, seq_len, patch_len, embed_dim, n_heads, n_layers, dropout=dropout)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)  # Weight decay
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)  # Learning rate decay

    # Training
    model.train()
    print("Training Progress:")
    for epoch in tqdm(range(num_epochs), desc="Epochs"):
        optimizer.zero_grad()
        for i in range(0, len(X_train_tensor), accumulation_steps):
            # Mini-batch
            X_batch = X_train_tensor[i:i + accumulation_steps].unsqueeze(-1)
            y_batch = y_train_tensor[i:i + accumulation_steps]

            # Forward pass
            outputs = model(X_batch)
            loss = criterion(outputs.squeeze(), y_batch)
            loss = loss / accumulation_steps  # Normalize loss for accumulation

            # Backward pass
            loss.backward()

            # Gradient accumulation step
            if (i + accumulation_steps) % accumulation_steps == 0 or i + accumulation_steps >= len(X_train_tensor):
                # Clip gradients to prevent explosion
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()

        # Update learning rate
        scheduler.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}")

    # Evaluation
    model.eval()
    y_pred_list = []
    print("\nEvaluation Progress:")
    with torch.no_grad():
        for i in tqdm(range(len(X_test_tensor)), desc="Evaluating"):
            y_pred = model(X_test_tensor[i].unsqueeze(0).unsqueeze(-1)).squeeze().item()
            y_pred_list.append(y_pred)
    
    # Prepare predictions for inverse transformation
    temp_pred = np.zeros((len(y_pred_list), data_scaled.shape[1]))  # Temporary array with same shape as original data
    temp_pred[:, 0] = y_pred_list  # Fill only the 'close' column (column 0)
    
    # Convert predictions and ground truth to actual values
    y_pred_actual = scaler.inverse_transform(temp_pred)[:, 0].flatten()  # Extract only the 'close' column after transformation
    y_test_actual = scaler.inverse_transform(
        np.column_stack((y_test_tensor.numpy(), np.zeros((len(y_test_tensor), 2))))  # Same adjustment for test targets
    )[:, 0].flatten()
    
    # Calculate metrics
    mse, rmse, mape, mpe = calculate_metrics(y_test_actual, y_pred_actual)

    return mse, rmse, mape, mpe

# Train and evaluate for close, SMA, and EMA
print("\nMetrics for Close Price:")
mse_close, rmse_close, mape_close, mpe_close = train_and_evaluate(
    X_train_tensor_close, y_train_tensor_close, X_test_tensor_close, y_test_tensor_close)
print(f"MSE: {mse_close}, RMSE: {rmse_close}, MAPE: {mape_close}%, MPE: {mpe_close}%")

print("\nMetrics for SMA:")
mse_sma, rmse_sma, mape_sma, mpe_sma = train_and_evaluate(
    X_train_tensor_sma, y_train_tensor_sma, X_test_tensor_sma, y_test_tensor_sma)
print(f"MSE: {mse_sma}, RMSE: {rmse_sma}, MAPE: {mape_sma}%, MPE: {mpe_sma}%")

print("\nMetrics for EMA:")
mse_ema, rmse_ema, mape_ema, mpe_ema = train_and_evaluate(
    X_train_tensor_ema, y_train_tensor_ema, X_test_tensor_ema, y_test_tensor_ema)
print(f"MSE: {mse_ema}, RMSE: {rmse_ema}, MAPE: {mape_ema}%, MPE: {mpe_ema}%")
