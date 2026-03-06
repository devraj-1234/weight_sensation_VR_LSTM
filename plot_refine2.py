import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import medfilt, savgol_filter
import os

def refine_vr_kinematics(csv_path):
    print(f"Loading and refining {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # 1. Normalize Time and Calculate precise dt (delta time)
    df['timestamp'] = df['timestamp'] - df['timestamp'].iloc[0]
    dt = np.mean(np.diff(df['timestamp']))
    fs = 1.0 / dt
    print(f"Detected exact sampling frequency: {fs:.2f} Hz")

    df_clean = df.copy()

    # 2. THE GOLDEN RULE: Clean Position First
    # Median filter kills 1-frame optical teleports
    med_pos_x = medfilt(df['pos_x'], kernel_size=5)
    med_pos_y = medfilt(df['pos_y'], kernel_size=5)
    med_pos_z = medfilt(df['pos_z'], kernel_size=5)

    # SavGol window of ~21 at 90Hz leaves the 8-12Hz muscle tremor INTACT, 
    # but smooths out the >20Hz optical static.
    window = 21 
    poly = 3
    df_clean['pos_x'] = savgol_filter(med_pos_x, window_length=window, polyorder=poly)
    df_clean['pos_y'] = savgol_filter(med_pos_y, window_length=window, polyorder=poly)
    df_clean['pos_z'] = savgol_filter(med_pos_z, window_length=window, polyorder=poly)

    # 3. Derive Linear Velocity from CLEAN Position
    # np.gradient calculates the exact calculus derivative
    df_clean['vel_x'] = np.gradient(df_clean['pos_x'], dt)
    df_clean['vel_y'] = np.gradient(df_clean['pos_y'], dt)
    df_clean['vel_z'] = np.gradient(df_clean['pos_z'], dt)

    # 4. Derive Linear Acceleration from CLEAN Velocity
    df_clean['acc_x'] = np.gradient(df_clean['vel_x'], dt)
    df_clean['acc_y'] = np.gradient(df_clean['vel_y'], dt)
    df_clean['acc_z'] = np.gradient(df_clean['vel_z'], dt)

    # 5. Handle Angular Velocity (Trust the Gyroscope, just remove spikes)
    df_clean['ang_vel_x'] = medfilt(df['ang_vel_x'], kernel_size=5)
    df_clean['ang_vel_y'] = medfilt(df['ang_vel_y'], kernel_size=5)
    df_clean['ang_vel_z'] = medfilt(df['ang_vel_z'], kernel_size=5)
    
    # 6. Recalculate Mechanical Power Proxy using the newly perfected data
    # P = a_x*v_x + (a_y + 9.81)*v_y + a_z*v_z
    df_clean['power'] = (df_clean['acc_x'] * df_clean['vel_x']) + \
                        ((df_clean['acc_y'] + 9.81) * df_clean['vel_y']) + \
                        (df_clean['acc_z'] * df_clean['vel_z'])

    # (Rotations rot_x, rot_y, rot_z, rot_w are left entirely untouched!)

    # Save the polished dataset
    # cleaned_filename = csv_path.replace('.csv', '_CLEANED.csv')
    # df_clean.to_csv(cleaned_filename, index=False)
    # print(f"Success! ML-ready data saved to: {cleaned_filename}")

    # --- VISUALIZATION ---
    fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
    fig.suptitle('Raw Unity Output vs. Derived Python Kinematics', fontsize=14, fontweight='bold')

    # Plot Position
    axes[0].plot(df['timestamp'], df['pos_y'], color='red', alpha=0.3, label='Raw Unity Position')
    axes[0].plot(df['timestamp'], df_clean['pos_y'], color='green', label='Filtered Position', linewidth=2)
    axes[0].set_ylabel('Position Y (m)')
    axes[0].legend()
    
    # Plot Velocity
    axes[1].plot(df['timestamp'], df['vel_y'], color='orange', alpha=0.3, label='Raw Unity Velocity')
    axes[1].plot(df['timestamp'], df_clean['vel_y'], color='green', label='Filtered Velocity', linewidth=2)
    axes[1].set_ylabel('Velocity Y (m/s)')
    axes[1].legend()

    # Plot Acceleration (The Moment of Truth)
    axes[2].plot(df['timestamp'], df['acc_y'], color='red', alpha=0.3, label='Raw Unity Acceleration (Noisy)')
    axes[2].plot(df['timestamp'], df_clean['acc_y'], color='blue', label='Derived Python Acceleration (Clean peaks!)', linewidth=1.5)
    axes[2].set_ylabel('Acceleration Y (m/s²)')
    axes[2].legend()

    # Plot Power
    axes[3].plot(df['timestamp'], df_clean['power'], color='purple', label='Re-calculated Power Proxy', linewidth=2)
    axes[3].set_ylabel('Power')
    axes[3].set_xlabel('Time (Seconds)')
    axes[3].legend()

    plt.tight_layout()
    plt.show()

# ==========================================
if __name__ == "__main__":
    target_file = "Data/Waste CSV 5th March/Telemetry_3kg_Grab1_230043.csv"  
    refine_vr_kinematics(target_file)