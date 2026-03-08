import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import medfilt, butter, filtfilt, savgol_filter
import os

def apply_vr_filters(data_array, fs, cutoff=5.0, median_window=5, sg_window=2, sg_poly=0):
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
    
    if sg_window % 2 == 0:
        sg_window += 1
    savgol_filtered = savgol_filter(
        med_filtered,
        window_length=sg_window,
        polyorder=sg_poly,
    )
    # window = 15
    # poly = 3
    # clean_data = savgol_filter(med_filtered, window_length=window, polyorder=poly)
    return med_filtered, savgol_filtered

def refine_and_plot_csv(target_csv, weight_label):
    if not os.path.exists(target_csv):
        print(f"Error: Could not find {target_csv}")
        return

    print(f"Loading {target_csv}...")
    df = pd.read_csv(target_csv)
    df_clean = df.copy()
    df_median = df.copy()
    df_savgol = df.copy()
    
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
        'pos_x', 'pos_y', 'pos_z'
    ]

    print("Applying Median and Median+Savgol filters...")
    for col in columns_to_process:
        med, sav = apply_vr_filters(df[col].values, fs)
        df_median[col] = med
        df_savgol[col] = sav
        
    for axis in ['x', 'y', 'z']:
        
         # RAW derivatives
        df['vel_' + axis] = np.gradient(df['pos_' + axis], dt)
        df['acc_' + axis] = np.gradient(df['vel_' + axis], dt)

        # MEDIAN derivatives
        df_median['vel_' + axis] = np.gradient(df_median['pos_' + axis], dt)
        df_median['acc_' + axis] = np.gradient(df_median['vel_' + axis], dt)

        # MEDIAN + SAVGOL derivatives
        df_savgol['vel_' + axis] = np.gradient(df_savgol['pos_' + axis], dt)
        df_savgol['acc_' + axis] = np.gradient(df_savgol['vel_' + axis], dt)
        
    for dataset in [df, df_median, df_savgol]:
        dataset['power'] = dataset['vel_y'] * (dataset['acc_y'] + 9.81)

    # --- SAVE THE CLEANED DATA ---
    # cleaned_filename = target_csv.replace('.csv', '_CLEANED_SAVGOL.csv')
    # df_clean.to_csv(cleaned_filename, index=False)
    # print(f"Success! Cleaned data saved to: {cleaned_filename}")

    # --- PLOT THE RESULTS ---
    # We will plot the Y-axis (Vertical) to visualize the heaviest part of the lift
    plot_cols = ['pos_y', 'vel_y', 'acc_y', 'power']
    titles = ['Vertical Position (m)', 'Vertical Velocity (m/s)', 'Vertical Acceleration (m/s²)', 'Mechanical Power']

    fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(10, 12), sharex=True)
    fig.suptitle('Raw vs. Refined VR Kinematics ' + str(weight_label) + ' kg', fontsize=16, fontweight='bold')

  
    for i, col in enumerate(plot_cols):
        # RAW
        axes[i].plot(
            df['timestamp'],
            df[col],
            color='red',
            alpha=0.5,
            label='Raw',
            linewidth=1.2
        )

        # MEDIAN
        axes[i].plot(
            df['timestamp'],
            df_median[col],
            color='blue',
            label='Median Filter',
            linewidth=2
        )

        # MEDIAN + SAVGOL
        axes[i].plot(
            df['timestamp'],
            df_savgol[col],
            color='green',
            label='Median + SavGol',
            linewidth=2.5
        )

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
    file_to_refine = "Data/curls 6th March/Telemetry_2kg_Grab1_215639.csv"  
    
    refine_and_plot_csv(file_to_refine, 2)