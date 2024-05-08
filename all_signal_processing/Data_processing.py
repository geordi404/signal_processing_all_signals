import os
import pandas as pd
import matplotlib.pyplot as plt

def Data_processing(directory, patient, session):
    full_session = f"{patient}_{session}"
    directory_path = os.path.join(directory, patient, full_session, f"{full_session}_aligned")

    csv_files = [
        f'{full_session}_biopac.csv',
        f'{full_session}_perclos_data.csv',
        f'{full_session}_simulator_data.csv'
    ]

    # Create a dictionary to hold dataframes
    dataframes = {}

    for csv_file in csv_files:
        file_path = os.path.join(directory_path, csv_file)
        file_exists = os.path.exists(file_path)
        print(f"Checking file {file_path}: Exists? {file_exists}")
        if file_exists:
            df = pd.read_csv(file_path)
            if not df.empty:
                df['time'] = pd.to_datetime(df['time'], unit='s')  # Convert 'time' to datetime
                df.set_index('time', inplace=True)  # Set 'time' as the index
                dataframes[csv_file] = df
            else:
                print(f"File is empty: {csv_file}")
        else:
            print(f"File not found: {csv_file}")

    if dataframes:
        # Define a common time index from the minimum start time to the maximum end time with a frequency of 2 ms
        start_time = max(df.index.min() for df in dataframes.values())
        end_time = min(df.index.max() for df in dataframes.values())
        common_index = pd.date_range(start=start_time, end=end_time, freq='2ms')

        # Resample and interpolate dataframes
        resampled_dfs = {}
        for name, df in dataframes.items():
            resampled_df = df.reindex(common_index, method='ffill')  # Use forward fill to interpolate
            resampled_dfs[name] = resampled_df
            print(f"Resampled and Interpolated {name}:")
            print(resampled_df.head())  # Print the first few rows of the resampled DataFrame

        plot_data(resampled_dfs, patient, session)
    else:
        print("No dataframes were loaded, check file paths and file content.")

    return dataframes  # Optional, if you want to use the resampled data elsewhere


def plot_data(resampled_dfs, patient, session):
    full_session = f"{patient}_{session}"
    simulator_key = f'{full_session}_simulator_data.csv'
    perclos_key = f'{full_session}_perclos_data.csv'

    try:
        simulator_df = resampled_dfs[simulator_key]
        perclos_df = resampled_dfs[perclos_key]
    except KeyError as e:
        print(f"Key not found in the dataset: {e}")
        return  # Exit the function if key is not found

    # Extract 'Road Position (m)' and 'Perclos'
    road_position = simulator_df['Road Position (m)']
    perclos = perclos_df['Perclos']

    # Calculate the rolling standard deviation for Road Position over a 1-minute window
    road_position_std = road_position.rolling(window=1*30000).std()

    # Create a single plot
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Plot Road Position Standard Deviation on the primary y-axis
    color = 'tab:red'
    ax1.plot(road_position_std.index, road_position_std, color=color)
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Standard Deviation of Road Position (m)', color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    # Create a second y-axis for Perclos
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.plot(perclos.index, perclos, color=color)
    ax2.set_ylabel('Perclos', color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    # Title and show the plot
    plt.title('Standard Deviation of Road Position and Perclos Over Time')
    plt.tight_layout()
    plt.show()