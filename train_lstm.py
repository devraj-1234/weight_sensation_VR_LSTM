import os
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# ==========================================
# 1. HYPERPARAMETERS & CONFIGURATION
# ==========================================
DATA_DIR = "Data"              # Folder containing your _CLEANED.csv files
SEQ_LENGTH = 30                # 30 frames of history (~0.33 seconds at 90Hz)
BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 0.001

# We use 18 features: 17 kinematic + 1 weight label
INPUT_SIZE = 18 
HIDDEN_SIZE = 64
NUM_LAYERS = 2
OUTPUT_SIZE = 3                # Predicting 3D Velocity (vel_x, vel_y, vel_z)

# Set device to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Training on device: {device}")

# ==========================================
# 2. DATA PREPARATION (The Sliding Window)
# ==========================================
class VRPseudoHapticDataset(Dataset):
    def __init__(self, data_dir, seq_length):
        self.seq_length = seq_length
        self.X = []
        self.Y = []
        
        # Look for all cleaned CSVs
        file_paths = glob.glob(os.path.join(data_dir, "*_CLEANED.csv"))
        if not file_paths:
            raise ValueError("No cleaned CSV files found. Check your directory!")

        all_features = []
        all_targets = []
        
        print("Loading and compiling datasets...")
        for file in file_paths:
            df = pd.read_csv(file)
            
            # The 18 Input Features
            features = df[['pos_x', 'pos_y', 'pos_z', 
                           'rot_x', 'rot_y', 'rot_z', 'rot_w',
                           'vel_x', 'vel_y', 'vel_z', 
                           'acc_x', 'acc_y', 'acc_z', 
                           'ang_vel_x', 'ang_vel_y', 'ang_vel_z', 
                           'power', 'weight_label']].values
            
            # The 3 Target Features (Velocity we want to predict)
            targets = df[['vel_x', 'vel_y', 'vel_z']].values
            
            all_features.append(features)
            all_targets.append(targets)

        # Concatenate everything to fit the StandardScaler globally
        combined_features = np.vstack(all_features)
        
        # We MUST scale neural network inputs so large numbers (power) 
        # don't drown out small numbers (quaternions)
        self.scaler = StandardScaler()
        combined_features_scaled = self.scaler.fit_transform(combined_features)
        
        # Re-split and build sliding windows
        current_idx = 0
        for i in range(len(all_features)):
            length = len(all_features[i])
            scaled_clip = combined_features_scaled[current_idx : current_idx + length]
            target_clip = all_targets[i]
            current_idx += length
            
            # Build the sequence windows
            for j in range(length - self.seq_length):
                # X is the history window [j : j + seq_length]
                window_x = scaled_clip[j : j + self.seq_length]
                
                # Spatial Normalization (Translation Invariance)
                # Zero out the position based on the first frame of the window
                window_x[:, 0:3] = window_x[:, 0:3] - window_x[0, 0:3]
                
                # Y is the actual velocity at the NEXT frame
                window_y = target_clip[j + self.seq_length]
                
                self.X.append(window_x)
                self.Y.append(window_y)

        self.X = torch.tensor(np.array(self.X), dtype=torch.float32)
        self.Y = torch.tensor(np.array(self.Y), dtype=torch.float32)
        print(f"Total Sequence Windows Generated: {len(self.X)}")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

# ==========================================
# 3. THE LSTM ARCHITECTURE
# ==========================================
class PseudoHapticLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(PseudoHapticLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # The core LSTM engine
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        
        # The fully connected output layer that maps the hidden state to 3D velocity
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # Initialize hidden state and cell state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        
        # Push through LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # We only care about the prediction at the very last time step of the window
        out = out[:, -1, :] 
        
        # Map to 3D velocity
        out = self.fc(out)
        return out

# ==========================================
# 4. TRAINING LOOP
# ==========================================
def train_model():
    dataset = VRPseudoHapticDataset(DATA_DIR, SEQ_LENGTH)
    
    # Split into 80% training, 20% validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    model = PseudoHapticLSTM(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE).to(device)
    
    # Mean Squared Error for regression physics tracking
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    train_losses = []
    val_losses = []

    print("\nStarting Training...")
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * batch_x.size(0)
            
        epoch_train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_train_loss)
        
        # Validation Loop
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                val_loss = criterion(outputs, batch_y)
                running_val_loss += val_loss.item() * batch_x.size(0)
                
        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        val_losses.append(epoch_val_loss)
        
        if (epoch+1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}] | Train Loss: {epoch_train_loss:.5f} | Val Loss: {epoch_val_loss:.5f}")

    # Save the trained model weights
    torch.save(model.state_dict(), "vr_haptic_lstm.pth")
    print("Model saved to vr_haptic_lstm.pth")

    # Plot the learning curve
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('LSTM Physics Learning Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Mean Squared Error Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    train_model()