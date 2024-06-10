import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import neurokit2 as nk
import math
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from scipy.signal import butter, filtfilt
from Calculate_new_perclos import Calculate_new_perclos

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y

def downsample_data(data, original_rate=500, target_rate=100):
    factor = original_rate // target_rate
    return data[::factor]

def smooth_transition(series, window_size):
    window_size = window_size * 500  # Convert window size to samples
    half_window = window_size // 2

    smoothed_series = series.copy()
    change_points = np.where(np.diff(series) != 0)[0] + 1

    for cp in change_points:
        start = max(0, cp - half_window)
        end = min(len(series), cp + half_window)
        if start < end and end <= len(series):
            x = np.arange(start, end)
            y = series[start:end]
            interpolated_values = np.linspace(series.iloc[start], series.iloc[end - 1], end - start)

            # Apply a weighted average to blend original and interpolated values
            for i in range(start, end):
                blend_weight = min((i - start) / half_window, (end - i) / half_window)
                smoothed_series.iloc[i] = (1 - blend_weight) * series.iloc[i] + blend_weight * interpolated_values[i - start]

    return smoothed_series

def Data_processing(directory, patient, session, Perclos_treshold, Perclos_window_size, Lane_deviation_window_size, plotting_activated):
    full_session = f"{patient}_{session}"
    directory_path = os.path.join(directory, patient, full_session, f"{full_session}_aligned")

    # Start with mandatory CSV files
    csv_files = [
        f'{full_session}_Biopac.csv',
        f'{full_session}_STM32ECG.csv',
        f'{full_session}_simulator_data.csv'
    ]

    DF_new_perclos = Calculate_new_perclos(directory, patient, session, Perclos_treshold, Perclos_window_size, 0)

    # Create a dictionary to hold dataframes
    dataframes = {}

    for csv_file in csv_files:
        file_path = os.path.join(directory_path, csv_file)
        file_exists = os.path.exists(file_path)
        print(f"Checking file {file_path}: Exists? {file_exists}")
        if file_exists:
            df = pd.read_csv(file_path)
            if not df.empty:
                df.set_index('time', inplace=True)  # Set 'time' as the index
                dataframes[csv_file] = df
            else:
                print(f"File is empty: {csv_file}")
        else:
            print(f"File not found: {csv_file}")

    if not DF_new_perclos.empty:
        DF_new_perclos.set_index('time', inplace=True)  # Set 'time' as the index
        dataframes['DF_new_perclos'] = DF_new_perclos
    else:
        print("DF_new_perclos DataFrame is empty")

    # Load Identifiant_route.csv
    identifiant_route_path = r'D:\IRSST_recordings\Recordings\Identifiant_route.csv'
    if os.path.exists(identifiant_route_path):
        identifiant_route_df = pd.read_csv(identifiant_route_path)
        id_to_sign = dict(zip(identifiant_route_df['Unique_ID'], identifiant_route_df['signe_route']))
        id_to_rayon = dict(zip(identifiant_route_df['Unique_ID'], identifiant_route_df['Rayon']))
    else:
        print(f"Identifiant_route.csv not found at {identifiant_route_path}")
        return 0

    if dataframes:
        # Define a common time index from the minimum start time to the maximum end time with a frequency of 2 ms
        start_time = max(df.index.min() for df in dataframes.values())
        end_time = min(df.index.max() for df in dataframes.values())
        common_index = np.arange(start_time, end_time, 0.002)

        # Resample and interpolate dataframes
        resampled_dfs = {}
        for name, df in dataframes.items():
            resampled_df = df.reindex(common_index, method='ffill')  # Use forward fill to interpolate
            resampled_dfs[name] = resampled_df

        # Add 'Direction' and 'Rayon' columns to simulator_df
        simulator_df = resampled_dfs[f'{full_session}_simulator_data.csv']
        simulator_df['Direction'] = simulator_df['Road ID'].map(id_to_sign)
        simulator_df['Rayon'] = simulator_df['Road ID'].map(id_to_rayon)
        simulator_df['Rayon_Signed'] = simulator_df['Rayon'] * simulator_df['Direction']

        # Identify missing mappings
        missing_keys = simulator_df[simulator_df['Direction'].isna()]['Road ID'].unique()
        if len(missing_keys) > 0:
            print("Missing keys in id_to_sign mapping:")
            print(missing_keys)

        # Calculate Ackermann angle in degrees multiplied by 2.85
        car_length = 3.962  # Length of the car in meters
        simulator_df['Ackermann_Angle'] = np.where(simulator_df['Road ID'] == 0, 0, np.degrees(np.arctan(car_length / simulator_df['Rayon_Signed'])) * 2.85)

        # Smooth the Ackermann angle with transition smoothing over 3 seconds
        simulator_df['Ackermann_Angle_Smoothed'] = smooth_transition(simulator_df['Ackermann_Angle'], 3)

        # Calculate Steering Wheel Compensated
        simulator_df['Steering_Wheel_Compensated'] = simulator_df['Steering Position'] + simulator_df['Ackermann_Angle_Smoothed']

        # Calculate the standard deviation of the smoothed steering wheel
        steering_wheel_std = simulator_df['Steering_Wheel_Compensated'].rolling(window=int(500 * Lane_deviation_window_size)).std()

        number_of_crash = plot_data_matplotlib(resampled_dfs, patient, session, Perclos_window_size, Lane_deviation_window_size, steering_wheel_std, plotting_activated)
    else:
        print("No dataframes were loaded, check file paths and file content.")

    return number_of_crash  # Optional, if you want to use the resampled data elsewhere

def plot_data_matplotlib(resampled_dfs, patient, session, Perclos_window_size, Lane_deviation_window_size, steering_wheel_std, plotting_activated):
    save = 0
    ackerman_angle_and_raw_steering_show = 1

    HALF_VEHICLE_WIDTH_LIST = [0, 0.838]
    plotting = int(plotting_activated)
 
    Number_of_crash = []

    ROAD_WIDTH = 3.3528
    full_session = f"{patient}_{session}"
    Road_accidents_events = []
    try:
        # Plotting lane deviation and Perclos
        for HALF_VEHICLE_WIDTH in HALF_VEHICLE_WIDTH_LIST:

            simulator_df = resampled_dfs[f'{full_session}_simulator_data.csv']
            perclos_df = resampled_dfs['DF_new_perclos']
            road_position = simulator_df['Road Position (m)']
            Road_position_accident = np.where((road_position > -HALF_VEHICLE_WIDTH) & (road_position <= (ROAD_WIDTH + HALF_VEHICLE_WIDTH)), 0, 1)
            # Create an array to store the result
            Road_position_accident_remove_extra_ones = np.zeros_like(Road_position_accident)

            # Find the segments of ones and keep only the first occurrence in each segment and replace the rest with zeroes
            in_accident_segment = False
            for i in range(len(Road_position_accident)):
                if Road_position_accident[i] == 1:
                    if not in_accident_segment:
                        Road_position_accident_remove_extra_ones[i] = 1
                        in_accident_segment = True
                else:
                    in_accident_segment = False

            # Append the number of crashes to the list
            Number_of_crash.append(np.sum(Road_position_accident_remove_extra_ones))
            Road_accidents_events.append(Road_position_accident_remove_extra_ones)
        
        perclos = perclos_df['new_perclos']
        road_position_std = road_position.rolling(window=int(500 * Lane_deviation_window_size)).std()
        if plotting == 1:
            # Create subplots with shared x-axis
            fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(16, 45), sharex=True)

            ax1.plot(road_position_std.index/60, road_position_std, color='tab:red')
            ax1.set_ylabel('Standard Deviation (m)', color='tab:red')
            ax1.tick_params(axis='y', labelcolor='tab:red')

            ax1b = ax1.twinx()
            ax1b.plot(perclos.index/60, perclos, color='tab:blue')
            ax1b.set_ylabel('Perclos', color='tab:blue')
            ax1b.tick_params(axis='y', labelcolor='tab:blue')
            

            # Plotting standard deviation of the smoothed steering wheel
            ax1c = ax1.twinx()
            ax1c.spines['right'].set_position(('outward', 60))
            ax1c.plot(steering_wheel_std.index/60, steering_wheel_std, color='tab:purple', label='Steering Wheel Compensated STD', linestyle='dashed')
            ax1c.set_ylabel('Steering Wheel Compensated STD')
            ax1c.tick_params(axis='y', labelcolor='tab:purple')
            ax1.set_title('Lane Deviation, Perclos, and Steering Wheel Compensated STD Over Time')

            if ackerman_angle_and_raw_steering_show == 1:
                # Plotting raw steering position and direction
                steering_position = simulator_df['Steering Position']
                direction = simulator_df['Direction']
                ax2.plot(steering_position.index/60, steering_position, color='tab:green', label='Steering Position')
                ax2.plot(direction.index/60, direction, color='tab:orange', label='Direction', linestyle='dotted')
                ax2.set_ylabel('Steering Position / Direction')
                ax2.legend(loc='upper right')
                ax2.set_title('Raw Steering Position and Direction Over Time')

                # Plotting Ackermann Angle and Smoothed Ackermann Angle
                ackermann_angle = simulator_df['Ackermann_Angle']
                ackermann_angle_smoothed = simulator_df['Ackermann_Angle_Smoothed']
                ax3.plot(ackermann_angle.index/60, ackermann_angle, color='tab:blue', label='Ackermann Angle (degrees)')
                ax3.plot(ackermann_angle_smoothed.index/60, ackermann_angle_smoothed, color='tab:red', label='Smoothed Ackermann Angle (degrees)')
                ax3.set_ylabel('Ackermann Angle (degrees)')
                ax3.legend(loc='upper right')
                ax3.set_title('Ackermann Angle and Smoothed Ackermann Angle Over Time')
            else:

                biopac_df = resampled_dfs[f'{full_session}_Biopac.csv']
                STM32_ECG= resampled_dfs[f'{full_session}_Biopac.csv']
                if 'Biopac_2' in biopac_df.columns:
                    ecg_signal = biopac_df['Biopac_2'].dropna()
                    ecg_signal 
                    # Apply bandpass filter to ECG signal
                    lowcut = 4
                    highcut = 40.0
                    fs = 500
                    filtered_ecg = bandpass_filter(ecg_signal, lowcut, highcut, fs)

                    processed_ecg = nk.ecg_process(filtered_ecg, sampling_rate=500)
                    r_peaks = processed_ecg[1]['ECG_R_Peaks']
                    rr_intervals = np.diff(r_peaks) * (1 / 500)
                    heart_rate = 60 / rr_intervals
                    hr_times = biopac_df.index[r_peaks[1:]]
                    hr_series = pd.Series(heart_rate, index=hr_times)
                    # Normalize the filtered ECG signal between 0 and 30

                    normalized_ecg = (filtered_ecg - np.min(filtered_ecg)) / (np.max(filtered_ecg) - np.min(filtered_ecg))
                    ax2.plot(hr_series.index/60, hr_series, color='tab:blue', label='Heart Rate (BPM)')
                    ax2.set_ylabel('Heart Rate (BPM)', color='tab:blue')
                    ax2.tick_params(axis='y', labelcolor='tab:blue')
                    ax2.legend(loc='upper right')
                    ax2.set_title('Heart Rate and Filtered ECG Signal Over Time')

                    # Create a secondary y-axis for the normalized ECG signal
                    ax2b = ax2.twinx()
                    ax2b.plot(biopac_df.index/60, normalized_ecg, color='tab:red', label='Filtered ECG Signal (Normalized)')
                    ax2b.set_ylabel('Filtered ECG Signal (Normalized)', color='tab:red')
                    ax2b.tick_params(axis='y', labelcolor='tab:red')
                    ax2b.legend(loc='upper left')

                # Plotting the main road position
                ax3.plot(road_position.index/60, road_position, color='tab:blue', label='road position (m)')
                Vehicle_portion = ["Half vehicle Crossing accident", "Full Vehicle Crossing accident"]
                color_accident = ['tab:red','tab:orange']
                # Loop to plot road accident events with dynamic labels and offset
                for i, Road_accident_event in enumerate(Road_accidents_events):
                    label = Vehicle_portion[i]
                    if i == 1:  # Apply offset to the second time series
                        Road_accident_event = Road_accident_event - 1.5
                    ax3.plot(road_position.index/60, Road_accident_event - 2, color=color_accident[i], label=label)

                ax3.set_ylabel('road_position (m)')
                ax3.legend(loc='upper right')
                ax3.set_title('Road position over time')

            # Plotting Steering Wheel Compensated
            steering_wheel_compensated = simulator_df['Steering_Wheel_Compensated']
            ax4.plot(steering_wheel_compensated.index/60, steering_wheel_compensated, color='tab:purple')
            ax4.set_ylabel('Steering position')
            ax4.set_title('Steering Wheel Compensated Over Time')
            plt.tight_layout(pad=8.0)
            plt.subplots_adjust(hspace=0.4)
            plt.show()

            # Saving the figure to a specified path
            if save == 1:
                save_path = "D:/recordings/data_analysis"
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                fig.savefig(f"{save_path}/{full_session}_analysis_figure.png", dpi=300)
        

    except KeyError as e:
        print(f"KeyError: {e}")
    except Exception as e:
        print(f"General error: {str(e)}")

    return Number_of_crash

def plot_data_html(resampled_dfs, patient, session, Perclos_window_size, Lane_deviation_window_size, steering_wheel_std, plotting_activated):
    save = 1  # Setting save to 1 to save the plots and data
    ackerman_angle_and_raw_steering_show = 0

    HALF_VEHICLE_WIDTH_LIST = [0, 0.838]
    plotting = int(plotting_activated)

    Number_of_crash = []

    ROAD_WIDTH = 3.3528
    full_session = f"{patient}_{session}"
    Road_accidents_events = []
    try:
        # Plotting lane deviation and Perclos
        for HALF_VEHICLE_WIDTH in HALF_VEHICLE_WIDTH_LIST:

            simulator_df = resampled_dfs[f'{full_session}_simulator_data.csv']
            perclos_df = resampled_dfs['DF_new_perclos']
            road_position = simulator_df['Road Position (m)']
            Road_position_accident = np.where((road_position > -HALF_VEHICLE_WIDTH) & (road_position <= (ROAD_WIDTH + HALF_VEHICLE_WIDTH)), 0, 1)
            # Create an array to store the result
            Road_position_accident_remove_extra_ones = np.zeros_like(Road_position_accident)

            # Find the segments of ones and keep only the first occurrence in each segment and replace the rest with zeroes
            in_accident_segment = False
            for i in range(len(Road_position_accident)):
                if Road_position_accident[i] == 1:
                    if not in_accident_segment:
                        Road_position_accident_remove_extra_ones[i] = 1
                        in_accident_segment = True
                else:
                    in_accident_segment = False

            # Append the number of crashes to the list
            Number_of_crash.append(np.sum(Road_position_accident_remove_extra_ones))
            Road_accidents_events.append(Road_position_accident_remove_extra_ones)
        
        perclos = perclos_df['new_perclos']
        road_position_std = road_position.rolling(window=int(500 * Lane_deviation_window_size)).std()
        if plotting == 1:
            # Downsample data for plotting to reduce size
            downsampled_time = downsample_data(road_position_std.index.values, 500, 100)
            downsampled_road_position_std = downsample_data(road_position_std.values, 500, 100)
            downsampled_perclos = downsample_data(perclos.values, 500, 100)
            downsampled_steering_wheel_std = downsample_data(steering_wheel_std.values, 500, 100)

            # Create Plotly figure with subplots
            fig = make_subplots(rows=5, cols=1, shared_xaxes=True, subplot_titles=(
                'Lane Deviation and Perclos', 'Heart Rate', 'Biopac ECG', 'Road Position', 'Steering Wheel Compensated'))

            # Add Lane Deviation and Perclos plot
            fig.add_trace(go.Scatter(x=downsampled_time / 60, y=downsampled_road_position_std, mode='lines', name='Standard Deviation (m)', line=dict(color='red')), row=1, col=1)
            fig.add_trace(go.Scatter(x=downsampled_time / 60, y=downsampled_perclos, mode='lines', name='Perclos', line=dict(color='blue')), row=1, col=1)

            # Add Steering Wheel Compensated STD plot
            fig.add_trace(go.Scatter(x=downsampled_time / 60, y=downsampled_steering_wheel_std, mode='lines', name='Steering Wheel Compensated STD', line=dict(color='purple', dash='dash')), row=1, col=1)

            biopac_df = resampled_dfs[f'{full_session}_Biopac.csv']
            if 'Biopac_2' in biopac_df.columns:
                ecg_signal = biopac_df['Biopac_2'].dropna()
                # Apply bandpass filter to ECG signal
                lowcut = 4
                highcut = 40.0
                fs = 500
                filtered_ecg = bandpass_filter(ecg_signal, lowcut, highcut, fs)

                # Downsample ECG data
                

                processed_ecg = nk.ecg_process(filtered_ecg, sampling_rate=500)
                r_peaks = processed_ecg[1]['ECG_R_Peaks']
                rr_intervals = np.diff(r_peaks) * (1 / 500)
                heart_rate = 60 / rr_intervals
                hr_times = biopac_df.index[r_peaks[1:]]
                hr_series = pd.Series(heart_rate, index=hr_times)
                # Normalize the filtered ECG signal between 0 and 30

                downsampled_ecg_signal = downsample_data(filtered_ecg, 500, 100)
                downsampled_ecg_time = downsample_data(biopac_df.index.values, 500, 100)
                
                normalized_ecg = (downsampled_ecg_signal - np.min(downsampled_ecg_signal)) / (np.max(downsampled_ecg_signal) - np.min(downsampled_ecg_signal))

                fig.add_trace(go.Scatter(x=hr_series.index / 60, y=hr_series, mode='lines', name='Heart Rate (BPM)', line=dict(color='green')), row=2, col=1)

                # Add Biopac ECG plot
                fig.add_trace(go.Scatter(x=downsampled_ecg_time / 60, y=downsampled_ecg_signal, mode='lines', name='Biopac ECG', line=dict(color='cyan')), row=3, col=1)

            if ackerman_angle_and_raw_steering_show == 1:
                # Add raw steering position and direction plot
                steering_position = simulator_df['Steering Position']
                direction = simulator_df['Direction']
                fig.add_trace(go.Scatter(x=steering_position.index / 60, y=steering_position, mode='lines', name='Steering Position', line=dict(color='green')), row=4, col=1)
                fig.add_trace(go.Scatter(x=direction.index, y=direction, mode='lines', name='Direction', line=dict(color='orange', dash='dot')), row=4, col=1)

                # Add Ackermann Angle and Smoothed Ackermann Angle plot
                ackermann_angle = simulator_df['Ackermann_Angle']
                ackermann_angle_smoothed = simulator_df['Ackermann_Angle_Smoothed']
                fig.add_trace(go.Scatter(x=ackermann_angle.index / 60, y=ackermann_angle, mode='lines', name='Ackermann Angle (degrees)', line=dict(color='blue')), row=5, col=1)
                fig.add_trace(go.Scatter(x=ackermann_angle_smoothed.index / 60, y=ackermann_angle_smoothed, mode='lines', name='Smoothed Ackermann Angle (degrees)', line=dict(color='red')), row=5, col=1)
            else:
                # Add road position plot
                fig.add_trace(go.Scatter(x=road_position.index / 60, y=road_position, mode='lines', name='Road Position (m)', line=dict(color='blue')), row=4, col=1)
                for i, Road_accident_event in enumerate(Road_accidents_events):
                    label = "Half vehicle Crossing accident" if i == 0 else "Full Vehicle Crossing accident"
                    color = 'red' if i == 0 else 'orange'
                    if i == 1:  # Apply offset to the second time series
                        Road_accident_event = Road_accident_event - 1.5
                    fig.add_trace(go.Scatter(x=road_position.index / 60, y=Road_accident_event - 2, mode='lines', name=label, line=dict(color=color)), row=4, col=1)

            # Add Steering Wheel Compensated plot
            steering_wheel_compensated = simulator_df['Steering_Wheel_Compensated']
            fig.add_trace(go.Scatter(x=steering_wheel_compensated.index / 60, y=steering_wheel_compensated, mode='lines', name='Steering Wheel Compensated', line=dict(color='purple')), row=5, col=1)

            fig.update_layout(height=1100, title_text=f"Patient {patient}, Session {session}")

            # Save the figure to an HTML file
            plot_html_path = os.path.join(r'D:\Recordings\data_analysis\Data_consulting', f'patient_{patient}_session_{session}.html')
            plotly_html = pio.to_html(fig, full_html=False)
            with open(plot_html_path, 'w', encoding='utf-8') as f:
                f.write(plotly_html)

            # Save all dataframes as CSV files
            base_directory = r'D:\Recordings\data_analysis\Data_consulting'
            for name, df in resampled_dfs.items():
                csv_path = os.path.join(base_directory, f'{name.replace(".csv", "")}_{full_session}.csv')
                df.to_csv(csv_path, index=True, encoding='utf-8')
        
    except KeyError as e:
        print(f"KeyError: {e}")
    except Exception as e:
        print(f"General error: {str(e)}")

    return Number_of_crash
