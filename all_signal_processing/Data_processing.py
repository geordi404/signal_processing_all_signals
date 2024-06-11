import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import neurokit2 as nk
import math
import plotly.graph_objs as go
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
from scipy.signal import butter, filtfilt
from Calculate_new_perclos import Calculate_new_perclos

def apply_iir_filter(data, fs, cutoff_freq, filter_type='low', order=4):
    """
    Apply an IIR filter to the input data.

    Parameters:
        data (array-like): Input data to be filtered.
        fs (float): Sampling frequency of the input data.
        cutoff_freq (float): Cutoff frequency of the filter.
        filter_type (str): Type of the filter ('low', 'high', 'bandpass', 'bandstop').
        order (int): Order of the filter.

    Returns:
        array-like: Filtered data.
    """
    if filter_type not in ['low', 'high', 'bandpass', 'bandstop']:
        raise ValueError("filter_type must be 'low', 'high', 'bandpass', or 'bandstop'")
    if not isinstance(order, int) or order < 1:
        raise ValueError("order must be a positive integer")
    if not isinstance(cutoff_freq,(list, np.ndarray)):
        raise ValueError("cutoff_freq must be a number")
    if not isinstance(fs, (int, float)):
        raise ValueError("fs must be a number")
    # if not isinstance(data, (list, np.ndarray)):
    #     raise ValueError("data must be an array-like object")
    cutoff_freq = np.array(cutoff_freq)
    nyquist_freq = 0.5 * fs
    normalized_cutoff_freq = cutoff_freq / nyquist_freq

    b, a = butter(order, normalized_cutoff_freq, btype=filter_type)

    filtered_data = filtfilt(b, a, data)
    
    return filtered_data

# Remove outliers
def remove_outliers(data, std_threshold=3):
    mean, std = np.mean(data), np.std(data)
    cut_off = std * std_threshold
    lower, upper = mean - cut_off, mean + cut_off
    data = np.clip(data, lower, upper)
    return data

def normalize_range(signal):
    norm_signal = (signal - np.min(signal)) / (np.max(signal) - np.min(signal))
    return 2 * norm_signal - 1

def preprocess_rsp_signal(rsp_signal, fs=500):               
    # Filter the respiration signal
    rsp_signal_filtered = apply_iir_filter(rsp_signal, fs, [0.05, 0.7], 'bandpass', 2)
    # rsp_signal_filtered = bandpass_filter(rsp_signal, 0.05, 0.7, fs, 2)
    
    # Find the peaks of the respiration signal
    rsp_signal_filtered_peaks = nk.rsp_findpeaks(rsp_signal_filtered, sampling_rate=fs)
    rsp_rate = nk.signal_rate(rsp_signal_filtered_peaks, desired_length=len(rsp_signal_filtered))
    
    # Normalize the respiration signal between -1 and 1
    rsp_signal = remove_outliers(rsp_signal)
    rsp_signal = normalize_range(rsp_signal)
    rsp_signal_filtered = remove_outliers(rsp_signal_filtered)
    rsp_signal_filtered = normalize_range(rsp_signal_filtered)
    
    return rsp_signal, rsp_signal_filtered, rsp_rate

def get_rsp_signals_ready(full_session, resampled_dfs, biopac_df):
    rsp_signal_0 = resampled_dfs[f'{full_session}_STM32RSP0.csv']
    rsp_signal_1 = resampled_dfs[f'{full_session}_STM32RSP1.csv']
    rsp_signal = biopac_df['Biopac_0'].dropna()
    rsp_signal_exp_0 = rsp_signal_0['Stm32RSP0_0'].dropna()
    rsp_signal_exp_1 = rsp_signal_0['Stm32RSP0_1'].dropna()
    rsp_signal_exp_2 = rsp_signal_1['Stm32RSP1_0'].dropna()
    rsp_signal_exp_3 = rsp_signal_1['Stm32RSP1_1'].dropna()
    rsp_signal, rsp_signal_filtered, rsp_rate = preprocess_rsp_signal(rsp_signal)
    rsp_signal_exp = []
    rsp_signal_exp_filtered = []
    rsp_signal_rate_exp = []
    signal, signal_exp_filtered, signal_rate_exp = preprocess_rsp_signal(rsp_signal_exp_0)
    rsp_signal_exp.append(signal)
    rsp_signal_exp_filtered.append(signal_exp_filtered)
    rsp_signal_rate_exp.append(signal_rate_exp)
    signal, signal_exp_filtered, signal_rate_exp = preprocess_rsp_signal(rsp_signal_exp_1)
    rsp_signal_exp.append(signal)
    rsp_signal_exp_filtered.append(signal_exp_filtered)
    rsp_signal_rate_exp.append(signal_rate_exp)
    signal, signal_exp_filtered, signal_rate_exp = preprocess_rsp_signal(rsp_signal_exp_2)
    rsp_signal_exp.append(signal)
    rsp_signal_exp_filtered.append(signal_exp_filtered)
    rsp_signal_rate_exp.append(signal_rate_exp)
    signal, signal_exp_filtered, signal_rate_exp = preprocess_rsp_signal(rsp_signal_exp_3)
    rsp_signal_exp.append(signal)
    rsp_signal_exp_filtered.append(signal_exp_filtered)
    rsp_signal_rate_exp.append(signal_rate_exp)
    return rsp_signal, rsp_signal_filtered, rsp_rate, rsp_signal_exp, rsp_signal_exp_filtered, rsp_signal_rate_exp

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

def Data_processing(directory, args_dict, patient, session, Perclos_treshold, Perclos_window_size, Lane_deviation_window_size, plotting_activated):
    full_session = f"{patient}_{session}"
    directory_path = os.path.join(directory, patient, full_session, f"{full_session}_aligned")

    # Start with mandatory CSV files
    csv_files = [
        f'{full_session}_Biopac.csv',
        f'{full_session}_STM32ECG.csv',
        f'{full_session}_simulator_data.csv',
        f'{full_session}_STM32RSP0.csv',
        f'{full_session}_STM32RSP1.csv',
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
    identifiant_route_path = r'E:\IRSST_recordings\Recordings\Identifiant_route.csv'
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
        # simulator_df['Ackermann_Angle_Smoothed'] = smooth_transition(simulator_df['Ackermann_Angle'], 3)
        simulator_df['Ackermann_Angle_Smoothed'] = simulator_df['Ackermann_Angle']

        # Calculate Steering Wheel Compensated
        simulator_df['Steering_Wheel_Compensated'] = simulator_df['Steering Position'] + simulator_df['Ackermann_Angle_Smoothed']

        # Calculate the standard deviation of the smoothed steering wheel
        steering_wheel_std = simulator_df['Steering_Wheel_Compensated'].rolling(window=int(500 * Lane_deviation_window_size)).std()

        if args_dict['plot_type'] == 'html':
            number_of_crash = plot_data_html(args_dict, resampled_dfs, patient, session, Perclos_window_size, Lane_deviation_window_size, steering_wheel_std, plotting_activated)
        else:
            number_of_crash = plot_data_matplotlib(args_dict, resampled_dfs, patient, session, Perclos_window_size, Lane_deviation_window_size, steering_wheel_std, plotting_activated)
    else:
        print("No dataframes were loaded, check file paths and file content.")

    return number_of_crash  # Optional, if you want to use the resampled data elsewhere

def plot_data_matplotlib(args_dict, resampled_dfs, patient, session, Perclos_window_size, Lane_deviation_window_size, steering_wheel_std, plotting_activated):
    save = 0
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
            diff_road_position_accident = np.diff(Road_position_accident)
            diff_road_position_accident[np.where(diff_road_position_accident == -1)[0]] = 0
            Road_position_accident_remove_extra_ones = np.insert(diff_road_position_accident, 0, Road_position_accident[0])
            
            # Append the number of crashes to the list
            Number_of_crash.append(np.sum(Road_position_accident_remove_extra_ones))
            Road_accidents_events.append(Road_position_accident_remove_extra_ones)
        
        perclos = perclos_df['new_perclos']
        road_position_std = road_position.rolling(window=int(500 * Lane_deviation_window_size)).std()
        if plotting == 1:
            # Create subplots with shared x-axis
            fig, (ax1, ax_rsp, ax_rsp_rate, ax3, ax4) = plt.subplots(5, 1, figsize=(16, 45), sharex=True)

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
                # ax2.plot(steering_position.index/60, steering_position, color='tab:green', label='Steering Position')
                # ax2.plot(direction.index/60, direction, color='tab:orange', label='Direction', linestyle='dotted')
                # ax2.set_ylabel('Steering Position / Direction')
                # ax2.legend(loc='upper right')
                # ax2.set_title('Raw Steering Position and Direction Over Time')

                # Plotting Ackermann Angle and Smoothed Ackermann Angle
                # ackermann_angle = simulator_df['Ackermann_Angle']
                # ackermann_angle_smoothed = simulator_df['Ackermann_Angle_Smoothed']
                # ax3.plot(ackermann_angle.index/60, ackermann_angle, color='tab:blue', label='Ackermann Angle (degrees)')
                # ax3.plot(ackermann_angle_smoothed.index/60, ackermann_angle_smoothed, color='tab:red', label='Smoothed Ackermann Angle (degrees)')
                # ax3.set_ylabel('Ackermann Angle (degrees)')
                # ax3.legend(loc='upper right')
                # ax3.set_title('Ackermann Angle and Smoothed Ackermann Angle Over Time')
            else:
                biopac_df = resampled_dfs[f'{full_session}_Biopac.csv']
                STM32_ECG= resampled_dfs[f'{full_session}_Biopac.csv']
                if ('Biopac_2' in biopac_df.columns) and (args_dict['ecg'] == '1'):
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
                    # ax2.plot(hr_series.index/60, hr_series, color='tab:blue', label='Heart Rate (BPM)')
                    # ax2.set_ylabel('Heart Rate (BPM)', color='tab:blue')
                    # ax2.tick_params(axis='y', labelcolor='tab:blue')
                    # ax2.legend(loc='upper right')
                    # ax2.set_title('Heart Rate and Filtered ECG Signal Over Time')

                    # Create a secondary y-axis for the normalized ECG signal
                    # ax2b = ax2.twinx()
                    # ax2b.plot(biopac_df.index/60, normalized_ecg, color='tab:red', label='Filtered ECG Signal (Normalized)')
                    # ax2b.set_ylabel('Filtered ECG Signal (Normalized)', color='tab:red')
                    # ax2b.tick_params(axis='y', labelcolor='tab:red')
                    # ax2b.legend(loc='upper left')
    
                if ('Biopac_0' in biopac_df.columns) & (args_dict['respiration'] == '1'):
                    rsp_signal, rsp_signal_filtered, rsp_rate, rsp_signal_exp, rsp_signal_exp_filtered, rsp_signal_rate_exp = get_rsp_signals_ready(full_session, resampled_dfs, biopac_df)
                    
                    # Plot respiration signal
                    ax_rsp.plot(rsp_signal.index/60, rsp_signal_filtered, label='Respiration Signal')
                    ax_rsp.set_ylabel('Respiration Signal' )
                    ax_rsp.set_xlabel('Time (minutes)')
                    ax_rsp.tick_params(axis='y')
                    ax_rsp.legend(loc='upper right')
                    ax_rsp.set_title('Respiration Signal Over Time')
 
                    ax_rsp_rate.plot(rsp_signal.index/60, rsp_rate, label='Respiration Rate')
                    ax_rsp_rate.set_ylabel('Respiration Rate')
                    ax_rsp_rate.tick_params(axis='y',)
                    
                    # Plot respiration signal for experimental data
                    for i in range(4):
                        ax_rsp.plot(rsp_signal_exp[i].index/60, rsp_signal_exp_filtered[i], label='Respiration Signal exp '+str(i+1))
                        ax_rsp.set_ylabel('Respiration Signal',)
                        ax_rsp.set_xlabel('Time (minutes)')
                        ax_rsp.tick_params(axis='y')
                        ax_rsp.legend(loc='upper right')
                        
                        ax_rsp_rate.plot(rsp_signal_exp[i].index/60, rsp_signal_rate_exp[i], label='Respiration Rate exp '+str(i+1))
                        ax_rsp_rate.legend(loc='upper left')
                    
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

def plot_data_html(args_dict, resampled_dfs, patient, session, Perclos_window_size, Lane_deviation_window_size, steering_wheel_std, plotting_activated):
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
            diff_road_position_accident = np.diff(Road_position_accident)
            diff_road_position_accident[np.where(diff_road_position_accident == -1)[0]] = 0
            Road_position_accident_remove_extra_ones = np.insert(diff_road_position_accident, 0, Road_position_accident[0])

            # Append the number of crashes to the list
            Number_of_crash.append(np.sum(Road_position_accident_remove_extra_ones))
            Road_accidents_events.append(Road_position_accident_remove_extra_ones)
        
        perclos = perclos_df['new_perclos']
        road_position_std = road_position.rolling(window=int(500 * Lane_deviation_window_size)).std()
        if plotting == 1:
            steering_wheel_compensated = simulator_df['Steering_Wheel_Compensated']
            # Downsample data for plotting to reduce size
            downsampled_time = downsample_data(road_position_std.index.values, 500, 100)
            downsampled_road_position_std = downsample_data(road_position_std.values, 500, 100)
            downsampled_perclos = downsample_data(perclos.values, 500, 100)
            downsampled_steering_wheel_std = downsample_data(steering_wheel_std.values, 500, 100)
            downsample_steering_wheel_compensated = downsample_data(steering_wheel_compensated.values, 500, 100)        

            # Create Plotly figure with subplots
            fig_rsp = make_subplots(rows=5, cols=1, shared_xaxes=True, subplot_titles=(
                'Lane Deviation and Perclos and steering wheel', 'Respiration Signal', 'Respiration rate', 'Road Position', 'PPG Signal'))

            # Add Lane Deviation and Perclos plot
            fig_rsp.add_trace(go.Scatter(x=downsampled_time / 60, y=downsampled_road_position_std, mode='lines', name='Standard Deviation (m)', line=dict(color='red')), row=1, col=1)
            fig_rsp.add_trace(go.Scatter(x=downsampled_time / 60, y=downsampled_perclos, mode='lines', name='Perclos', line=dict(color='blue')), row=1, col=1)

            # Add Steering Wheel Compensated STD plot
            fig_rsp.add_trace(go.Scatter(x=downsampled_time / 60, y=downsample_steering_wheel_compensated, mode='lines', name='Steering Wheel Compensated', line=dict(color='purple')), row=1, col=1)
            # fig_rsp.add_trace(go.Scatter(x=downsampled_time / 60, y=downsampled_steering_wheel_std, mode='lines', name='Steering Wheel Compensated STD', line=dict(color='purple', dash='dash')), row=1, col=1)
            print("Finished plotting Lane Deviation and Perclos")
            
            biopac_df = resampled_dfs[f'{full_session}_Biopac.csv']
            # if 'Biopac_2' in biopac_df.columns:
            #     ecg_signal = biopac_df['Biopac_2'].dropna()
            #     # Apply bandpass filter to ECG signal
            #     lowcut = 4
            #     highcut = 40.0
            #     fs = 500
            #     filtered_ecg = bandpass_filter(ecg_signal, lowcut, highcut, fs)

            #     # Downsample ECG data
            #     processed_ecg = nk.ecg_process(filtered_ecg, sampling_rate=500)
            #     r_peaks = processed_ecg[1]['ECG_R_Peaks']
            #     rr_intervals = np.diff(r_peaks) * (1 / 500)
            #     heart_rate = 60 / rr_intervals
            #     hr_times = biopac_df.index[r_peaks[1:]]
            #     hr_series = pd.Series(heart_rate, index=hr_times)
            #     # Normalize the filtered ECG signal between 0 and 30

            #     downsampled_ecg_signal = downsample_data(filtered_ecg, 500, 100)
            #     downsampled_ecg_time = downsample_data(biopac_df.index.values, 500, 100)
                
            #     normalized_ecg = (downsampled_ecg_signal - np.min(downsampled_ecg_signal)) / (np.max(downsampled_ecg_signal) - np.min(downsampled_ecg_signal))

            #     fig.add_trace(go.Scatter(x=hr_series.index / 60, y=hr_series, mode='lines', name='Heart Rate (BPM)', line=dict(color='green')), row=2, col=1)

            #     # Add Biopac ECG plot
            #     fig.add_trace(go.Scatter(x=downsampled_ecg_time / 60, y=downsampled_ecg_signal, mode='lines', name='Biopac ECG', line=dict(color='cyan')), row=3, col=1)
            #     print("Finished plotting Heart Rate and Biopac ECG")
            if ('Biopac_3' in biopac_df.columns):
                ppg_signal = biopac_df['Biopac_3'].dropna()
                ppg_signal = bandpass_filter(ppg_signal, 0.5, 6, 500, 2)
                # remove outliers
                ppg_signal = remove_outliers(ppg_signal, 0.6)
                # downsample ppg data
                downsampled_ppg_signal = downsample_data(ppg_signal, 500, 100)
                downsampled_ppg_time = downsample_data(biopac_df.index.values, 500, 100)
                # plot ppg signal
                fig_rsp.add_trace(go.Scatter(x=downsampled_ppg_time / 60, y=downsampled_ppg_signal, mode='lines', name='PPG Signal', line=dict(color='cyan')), row=5, col=1)
                print("Finished plotting PPG Signal")
            if ('Biopac_0' in biopac_df.columns) & (args_dict['respiration'] == '1'):
                rsp_signal, rsp_signal_filtered, rsp_rate, rsp_signal_exp, rsp_signal_exp_filtered, rsp_signal_rate_exp = get_rsp_signals_ready(full_session, resampled_dfs, biopac_df)
                downsample_rsp_signal = downsample_data(rsp_signal, 500, 100)
                downsample_rsp_signal_filtered = downsample_data(rsp_signal_filtered, 500, 100)
                downsample_rsp_rate = downsample_data(rsp_rate, 500, 100)
                downsample_rsp_time = downsample_data(rsp_signal.index.values, 500, 100)
                downsample_rsp_rate_exp = []
                downsample_rsp_signal_exp_filtered = []
                for i in range(4):
                    downsample_rsp_signal_exp = downsample_data(rsp_signal_exp[i], 500, 100)
                    data = downsample_data(rsp_signal_exp_filtered[i], 500, 100)
                    downsample_rsp_signal_exp_filtered.append(data)
                    data = downsample_data(rsp_signal_rate_exp[i], 500, 100)
                    downsample_rsp_rate_exp.append(data)
                # Plot respiration signal
                fig_rsp.add_trace(go.Scatter(x=downsample_rsp_time / 60, y=downsample_rsp_signal_filtered, mode='lines', name='Respiration Signal', line=dict(color='green')), row=2, col=1)
                fig_rsp.add_trace(go.Scatter(x=downsample_rsp_time / 60, y=downsample_rsp_rate, mode='lines', name='Respiration Rate', line=dict(color='green')), row=3, col=1)
                # Plot respiration signal for experimental data
                # Init 4 colors
                colors = ['orange', 'purple', 'brown', 'pink']
                for i in range(4):
                    fig_rsp.add_trace(go.Scatter(x=downsample_rsp_time / 60, y=downsample_rsp_signal_exp_filtered[i], mode='lines', name=f'Respiration Signal exp {i+1}', line=dict(color=colors[i])), row=2, col=1)
                    fig_rsp.add_trace(go.Scatter(x=downsample_rsp_time / 60, y=downsample_rsp_rate_exp[i], mode='lines', name=f'Respiration Rate exp {i+1}', line=dict(color=colors[i])), row=3, col=1)
                print("Finished plotting Respiration Signal and Respiration Rate")
            if ackerman_angle_and_raw_steering_show == 1:
                # Add raw steering position and direction plot
                steering_position = simulator_df['Steering Position']
                direction = simulator_df['Direction']
                fig_rsp.add_trace(go.Scatter(x=steering_position.index / 60, y=steering_position, mode='lines', name='Steering Position', line=dict(color='green')), row=4, col=1)
                fig_rsp.add_trace(go.Scatter(x=direction.index, y=direction, mode='lines', name='Direction', line=dict(color='orange', dash='dot')), row=4, col=1)

                # Add Ackermann Angle and Smoothed Ackermann Angle plot
                ackermann_angle = simulator_df['Ackermann_Angle']
                ackermann_angle_smoothed = simulator_df['Ackermann_Angle_Smoothed']
                fig_rsp.add_trace(go.Scatter(x=ackermann_angle.index / 60, y=ackermann_angle, mode='lines', name='Ackermann Angle (degrees)', line=dict(color='blue')), row=4, col=1)
                fig_rsp.add_trace(go.Scatter(x=ackermann_angle_smoothed.index / 60, y=ackermann_angle_smoothed, mode='lines', name='Smoothed Ackermann Angle (degrees)', line=dict(color='red')), row=6, col=1)
                print("Finished plotting Raw Steering Position, Direction, Ackermann Angle, and Smoothed Ackermann Angle")
            else:
                # # Add road position plot
                # downsample road position
                downsample_road_position = downsample_data(road_position, 500, 100)
                fig_rsp.add_trace(go.Scatter(x=downsampled_time / 60, y=downsample_road_position, mode='lines', name='Road Position (m)', line=dict(color='blue')), row=4, col=1)
                for i, Road_accident_event in enumerate(Road_accidents_events):
                    label = "Half vehicle Crossing accident" if i == 0 else "Full Vehicle Crossing accident"
                    color = 'red' if i == 0 else 'orange'
                    if i == 1:  # Apply offset to the second time series
                        Road_accident_event = Road_accident_event - 1.5
                    fig_rsp.add_trace(go.Scatter(x=road_position.index / 60, y=Road_accident_event - 2, mode='lines', name=label, line=dict(color=color)), row=4, col=1)
                print("Finished plotting Road Position")

            fig_rsp.update_layout(height=1100, title_text=f"Patient {patient}, Session {session}")

            # Save the figure to an HTML file
            plot_html_path = os.path.join(r'data_consulting', f'patient_{patient}_session_{session}.html')
            plotly_html = pio.to_html(fig_rsp, full_html=False)
            with open(plot_html_path, 'w', encoding='utf-8') as f:
                f.write(plotly_html)

            # Save all dataframes as CSV files
            # base_directory = r'data_consulting'
            # for name, df in resampled_dfs.items():
            #     csv_path = os.path.join(base_directory, f'{name.replace(".csv", "")}_{full_session}.csv')
            #     df.to_csv(csv_path, index=True, encoding='utf-8')
        
    except KeyError as e:
        print(f"KeyError: {e}")
    except Exception as e:
        print(f"General error: {str(e)}")

    return Number_of_crash
