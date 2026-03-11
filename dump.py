import os
import glob
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Make sure this matches where your CSVs are!
DATA_DIR = "Data/Cleaned_Data" 

print("Calculating scaler arrays...")

# 1. Find the files
search_pattern = os.path.join(DATA_DIR, "*_CLEANED_MEDIAN.csv")
file_paths = glob.glob(search_pattern)

if not file_paths:
    print(f"Error: Could not find CSVs in {DATA_DIR}")
    exit()

all_features = []

# 2. Load the data
for file in file_paths:
    df = pd.read_csv(file)
    features = df[['pos_x', 'pos_y', 'pos_z', 
                   'rot_x', 'rot_y', 'rot_z', 'rot_w',
                   'vel_x', 'vel_y', 'vel_z', 
                   'acc_x', 'acc_y', 'acc_z', 
                   'ang_vel_x', 'ang_vel_y', 'ang_vel_z', 
                   'power', 'weight_label']].values
    all_features.append(features)

# 3. Fit the scaler
combined_features = np.vstack(all_features)
scaler = StandardScaler()
scaler.fit(combined_features)

# 4. Print the C# code
print("\n--- COPY THESE INTO C# ---")
print(f"private float[] scalerMeans = new float[] {{ {', '.join(map(str, scaler.mean_))} }};")
print(f"private float[] scalerScales = new float[] {{ {', '.join(map(str, scaler.scale_))} }};")
print("--------------------------\n")