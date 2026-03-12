import os
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from tqdm import tqdm
import joblib

# ==========================================
# 1. HYPERPARAMETERS
# ==========================================

DATA_DIR = "Data/Cleaned_Data"
SEQ_LENGTH = 30
BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 0.001

INPUT_SIZE = 18
HIDDEN_SIZE = 64
NUM_LAYERS = 2
OUTPUT_SIZE = 3   # Predicting acceleration (ax, ay, az)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Training on:", device)

# ==========================================
# 2. DATASET
# ==========================================

class VRPseudoHapticDataset(Dataset):

    def __init__(self, file_paths, seq_length, scaler=None):

        self.seq_length = seq_length
        self.X = []
        self.Y = []

        all_features = []
        all_targets = []

        print(f"\nLoading {len(file_paths)} files")

        for file in file_paths:

            df = pd.read_csv(file)

            features = df[['pos_x','pos_y','pos_z',
                           'rot_x','rot_y','rot_z','rot_w',
                           'vel_x','vel_y','vel_z',
                           'acc_x','acc_y','acc_z',
                           'ang_vel_x','ang_vel_y','ang_vel_z',
                           'power','weight_label']].values

            # TARGET = ACCELERATION
            targets = df[['acc_x','acc_y','acc_z']].values

            all_features.append(features)
            all_targets.append(targets)

        combined_features = np.vstack(all_features)

        if scaler is None:
            self.scaler = StandardScaler()
            combined_features_scaled = self.scaler.fit_transform(combined_features)
        else:
            self.scaler = scaler
            combined_features_scaled = self.scaler.transform(combined_features)

        current_idx = 0

        print("\nBuilding sequence windows...")

        for i in tqdm(range(len(all_features)), desc="Processing Files"):

            length = len(all_features[i])

            scaled_clip = combined_features_scaled[current_idx: current_idx + length]
            target_clip = all_targets[i]

            current_idx += length

            for j in range(length - seq_length):

                window_x = scaled_clip[j:j+seq_length]

                # Make positions relative
                window_x[:,0:3] = window_x[:,0:3] - window_x[0,0:3]

                window_y = target_clip[j+seq_length]

                self.X.append(window_x)
                self.Y.append(window_y)

        self.X = torch.tensor(np.array(self.X), dtype=torch.float32)
        self.Y = torch.tensor(np.array(self.Y), dtype=torch.float32)

        print("Total windows:", len(self.X))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

# ==========================================
# 3. MODEL
# ==========================================

class PseudoHapticLSTM(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, output_size):

        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=0.2
        )

        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):

        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        out, _ = self.lstm(x, (h0, c0))

        out = out[:, -1, :]

        acceleration = self.fc(out)

        return acceleration

# ==========================================
# 4. TRAINING
# ==========================================

def train_model():

    file_paths = glob.glob(os.path.join(DATA_DIR, "*_CLEANED_MEDIAN.csv"))

    np.random.shuffle(file_paths)

    split_idx = int(0.8 * len(file_paths))

    train_files = file_paths[:split_idx]
    val_files = file_paths[split_idx:]

    train_dataset = VRPseudoHapticDataset(train_files, SEQ_LENGTH)
    scaler = train_dataset.scaler

    val_dataset = VRPseudoHapticDataset(val_files, SEQ_LENGTH, scaler)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    model = PseudoHapticLSTM(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_loss = float('inf')

    train_losses = []
    val_losses = []

    print("\nStarting training...")

    for epoch in range(EPOCHS):

        model.train()

        running_loss = 0

        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} Train")

        for batch_x, batch_y in train_bar:

            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()

            outputs = model(batch_x)

            loss = criterion(outputs, batch_y)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

            running_loss += loss.item() * batch_x.size(0)

        epoch_train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_train_loss)

        model.eval()

        running_val_loss = 0

        with torch.no_grad():

            for batch_x, batch_y in val_loader:

                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)

                outputs = model(batch_x)

                val_loss = criterion(outputs, batch_y)

                running_val_loss += val_loss.item() * batch_x.size(0)

        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        val_losses.append(epoch_val_loss)

        print(f"Epoch [{epoch+1}/{EPOCHS}] Train: {epoch_train_loss:.6f} Val: {epoch_val_loss:.6f}")

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), "best_vr_haptic_lstm.pth")

    print("\nBest validation loss:", best_val_loss)

    joblib.dump(scaler, "feature_scaler.pkl")
    print("Scaler saved")

    plt.plot(train_losses,label="Train Loss")
    plt.plot(val_losses,label="Val Loss")

    plt.title("LSTM Learning Curve")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")

    plt.legend()
    plt.grid()
    plt.show()


if __name__ == "__main__":
    train_model()