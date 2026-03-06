import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import medfilt, butter, filtfilt, savgol_filter
import os

def apply_vr_filters(data_array, fs, cutoff=5.0, median_window=5):
    """
    Applies the Two-Stage Biomechanical Filter.
    """
    '''
    # Step 1: Median Filter for optical tracking teleports/spikes
    med_filtered = medfilt(data_array, kernel_size=median_window)
    
    # Step 2: Zero-phase Butterworth Low-Pass Filter for physiological tremor
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(4, normal_cutoff, btype='low', analog=False)
    
    # filtfilt ensures zero phase shift (no temporal lag in the cleaned data)
    clean_data = filtfilt(b, a, med_filtered)
    '''
    med_filtered = medfilt(data_array, kernel_size=median_window)
    clean_data = med_filtered
    # window = 15
    # poly = 3
    # clean_data = savgol_filter(med_filtered, window_length=window, polyorder=poly)
    return clean_data

def refine_and_plot_csv(target_csv):
    if not os.path.exists(target_csv):
        print(f"Error: Could not find {target_csv}")
        return

    print(f"Loading {target_csv}...")
    df = pd.read_csv(target_csv)
    
    # Normalize time to start at 0.0
    df['timestamp'] = df['timestamp'] - df['timestamp'].iloc[0]
    
    # --- DYNAMIC SAMPLING FREQUENCY ---
    # Calculate average time delta between frames to find exact Hz
    dt = np.mean(np.diff(df['timestamp']))
    fs = 1.0 / dt
    print(f"Detected exact sampling frequency: {fs:.2f} Hz")

    # --- COLUMNS TO FILTER ---
    # We filter position, velocity, acceleration, angular velocity, and power.
    # We DO NOT filter Quaternions (rot_x, rot_y, rot_z, rot_w) to preserve valid 3D rotations.
    columns_to_process = [
        'pos_x', 'pos_y', 'pos_z', 
        'vel_x', 'vel_y', 'vel_z', 
        'acc_x', 'acc_y', 'acc_z', 
        'ang_vel_x', 'ang_vel_y', 'ang_vel_z', 
        'power'
    ]

    df_clean = df.copy()

    print("Applying Median and Butterworth filters...")
    for col in columns_to_process:
        df_clean[col] = apply_vr_filters(df[col].values, fs=fs, cutoff=5.0, median_window=5)

    # --- SAVE THE CLEANED DATA ---
    cleaned_filename = target_csv.replace('.csv', '_CLEANED_SAVGOL.csv')
    # df_clean.to_csv(cleaned_filename, index=False)
    # print(f"Success! Cleaned data saved to: {cleaned_filename}")

    # --- PLOT THE RESULTS ---
    # We will plot the Y-axis (Vertical) to visualize the heaviest part of the lift
    plot_cols = ['pos_y', 'vel_y', 'acc_y', 'power']
    titles = ['Vertical Position (m)', 'Vertical Velocity (m/s)', 'Vertical Acceleration (m/s²)', 'Mechanical Power']

    fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(10, 12), sharex=True)
    fig.suptitle('Raw vs. Refined VR Kinematics', fontsize=16, fontweight='bold')

    for i, col in enumerate(plot_cols):
        # Plot Raw (Red, slightly transparent)
        axes[i].plot(df['timestamp'], df[col], color='red', alpha=0.5, label='Raw (Jitter + Glitches)', linewidth=1.5)
        
        # Plot Cleaned (Green, bold)
        axes[i].plot(df['timestamp'], df_clean[col], color='green', label='Refined (Filtered)', linewidth=2.5)
        
        axes[i].set_ylabel(titles[i])
        axes[i].legend(loc='upper right')
        axes[i].grid(True, linestyle='--', alpha=0.7)

    axes[-1].set_xlabel('Time (Seconds)')
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    
    
    
    plt.show()

# ==========================================
# EXECUTION
# ==========================================
if __name__ == "__main__":
    # Replace this with the actual name of your file in the Data folder!
    # Tip: Use your 0kg bicep curl file to see it fix the tracking spikes.
    file_to_refine = "Data/Waste CSV 5th March/Telemetry_3kg_Grab2_230107.csv" 
    
    refine_and_plot_csv(file_to_refine)