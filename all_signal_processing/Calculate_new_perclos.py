import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def Calculate_new_perclos(directory, patient, session, Perclos_threshold, sessionPerclos_window_size, plotting_perclos):
    full_session = f"{patient}_{session}"
    directory_path = os.path.join(directory, patient, full_session, f"{full_session}_aligned")
    csv_file = f'{full_session}_perclos_data.csv'
    file_path = os.path.join(directory_path, csv_file)

    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_path)
    #print(df)
    # Consider the eye closed when the average of ear_l and ear_r are 25% of the max_average
    closure_threshold = Perclos_threshold
    
    # Calculate the new perclos
    df['avg_ear'] = (df['ear_l'] + df['ear_r']) / 2
    df['eye_closed'] = df['avg_ear'] <= closure_threshold
    
    # Calculate the percentage of eye closure using a rolling 1-minute window (1800 samples)
    rolling_perclos = df['eye_closed'].rolling(window=30 * sessionPerclos_window_size, min_periods=1).sum() / (30 * sessionPerclos_window_size)
    
    # Use the existing perclos for the first minute
    new_perclos_first_minute = df['perclos'].iloc[:30 * sessionPerclos_window_size]
    
    # Concatenate the first minute of perclos with the new rolling perclos
    new_perclos_rest = rolling_perclos.iloc[30 * sessionPerclos_window_size:]
    df['new_perclos'] = pd.concat([new_perclos_first_minute, new_perclos_rest])
    
    if plotting_perclos == 1:
        # Plot original and new perclos
        plt.figure(figsize=(14, 7))
        plt.plot(df['time'], df['perclos'], label='Original PERCLOS', color='blue')
        plt.plot(df['time'], df['new_perclos'], label='New PERCLOS', color='red')
        plt.xlabel('Time')
        plt.ylabel('PERCLOS')
        plt.title(f'PERCLOS Comparison for {full_session}')
        plt.legend()
        plt.grid(True)
        plt.show()

    # Save the updated DataFrame back to the CSV file
        #df.to_csv(file_path, index=False)
    
    return df
