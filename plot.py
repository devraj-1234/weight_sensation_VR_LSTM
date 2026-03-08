import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def refine_and_plot_csv(target_csv, weight_label):
    if not os.path.exists(target_csv):
        print(f"Error: Could not find {target_csv}")
        return

    print(f"Loading {target_csv}...")
    df = pd.read_csv(target_csv)
    df_clean = df.copy()
    
    filtered_csv = target_csv.replace('.csv', '_CLEANED_MEDIAN.csv')
    if not os.path.exists(filtered_csv):
        print(f"Error: Could not find the cleaned CSV file: {filtered_csv}")
        return
    print(f"Loading cleaned data from {filtered_csv}...")
    df_median = pd.read_csv(filtered_csv)
    
    
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
            df_clean[col],
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
    file_to_refine = "Data/curls 6th March/Telemetry_4kg_Grab1_230219.csv"  
    
    refine_and_plot_csv(file_to_refine, 4)