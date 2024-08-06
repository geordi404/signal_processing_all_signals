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
from scipy.signal import butter, filtfilt, find_peaks
import pdb  
from scipy.interpolate import interp1d
import heartpy as hp
from scipy.signal import resample
import pickle
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from scipy.stats import kurtosis, skew
from tqdm import tqdm
from scipy.signal import welch, iirnotch, filtfilt
from filelock import FileLock, Timeout
import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

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

def Data_processing(directory, patient, session, Perclos_treshold, Perclos_window_size, Lane_deviation_window_size, plotting_activated,Drowsiness_Accuracy_signal_quality_csv_total,Drowsiness_Accuracy_signal_quality_csv_granularized):
    full_session = f"{patient}_{session}"
    directory_path = os.path.join(directory, patient, full_session, f"{full_session}_aligned")

    # Start with mandatory CSV files
    csv_files = [
        f'{full_session}_Biopac.csv',
        f'{full_session}_Stm32ECG.csv',
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
    identifiant_route_path = r'E:\Recordings\Identifiant_route.csv'
    if os.path.exists(identifiant_route_path):
        identifiant_route_df = pd.read_csv(identifiant_route_path)
        id_to_sign = dict(zip(identifiant_route_df['Unique_ID'], identifiant_route_df['signe_route']))
        id_to_rayon = dict(zip(identifiant_route_df['Unique_ID'], identifiant_route_df['Rayon']))
        print("id_to_sign")
        print(id_to_sign)
        print("id_to_rayon")
        print(id_to_rayon)
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

        #number_of_crash = plot_data_PERCLOS_LANE_DEVIATION(resampled_dfs, patient, session, Perclos_window_size, Lane_deviation_window_size, steering_wheel_std, plotting_activated)
        #plot_ecg_signals(resampled_dfs, patient, session, plotting_activated)
        #plot_ppg_signal(resampled_dfs, patient, session, plotting_activated)
        plot_ecg_signals_V_2_0(directory,resampled_dfs, patient, session, plotting_activated,Drowsiness_Accuracy_signal_quality_csv_total,Drowsiness_Accuracy_signal_quality_csv_granularized)
    else:
        print("No dataframes were loaded, check file paths and file content.")

    #return number_of_crash  # Optional, if you want to use the resampled data elsewhere

def plot_data_PERCLOS_LANE_DEVIATION(resampled_dfs, patient, session, Perclos_window_size, Lane_deviation_window_size, steering_wheel_std, plotting_activated):
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
            
            # Determine if each road position is within the safe range or not
            Road_position_accident = np.where(
                (road_position > -HALF_VEHICLE_WIDTH) & (road_position <= (ROAD_WIDTH + HALF_VEHICLE_WIDTH)), 
                0, 
                1
            )
            
            # Detect transitions from 0 to 1
            transitions = np.diff(Road_position_accident, prepend=0)
            
            # Qualify the 0 to 1 transitions and ignore others
            Road_position_accident_remove_extra_ones = np.where(transitions == 1, 1, 0)
            
            # Ensure the result has the same length as the original road_position array
            if len(Road_position_accident_remove_extra_ones) < len(Road_position_accident):
                # Append a 0 at the end to match the size
                Road_position_accident_remove_extra_ones = np.append(Road_position_accident_remove_extra_ones, 0)
            elif len(Road_position_accident_remove_extra_ones) > len(Road_position_accident):
                # Truncate the last element to match the size
                Road_position_accident_remove_extra_ones = Road_position_accident_remove_extra_ones[:len(Road_position_accident)]
                    # Append the number of crashes to the list
                    
            Number_of_crash.append(np.sum(Road_position_accident_remove_extra_ones))
            Road_accidents_events.append(Road_position_accident_remove_extra_ones)
        
        perclos = perclos_df['new_perclos']
        road_position_std = road_position.rolling(window=int(500 * Lane_deviation_window_size)).std()
        if plotting == 1:
            # Create subplots with shared x-axis
            fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 45), sharex=True)

            print(road_position_std.index)
            ax1.plot(road_position_std.index/60, road_position_std, color='tab:red')
            ax1.set_ylabel('Lane deviation STD (m)', color='tab:red', fontsize=8)
            ax1.set_xlabel('Time', fontsize=8)
            ax1.tick_params(axis='y', labelcolor='tab:red')

            ax1b = ax1.twinx()
            ax1b.plot(perclos.index/60, perclos, color='tab:blue')
            ax1b.set_ylabel('Perclos', color='tab:blue', fontsize=8)
            ax1b.tick_params(axis='y', labelcolor='tab:blue')
            

            # Plotting standard deviation of the smoothed steering wheel
            ax1c = ax1.twinx()
            ax1c.spines['right'].set_position(('outward', 60))
            ax1c.plot(steering_wheel_std.index/60, steering_wheel_std, color='tab:purple', label='Steering Wheel Compensated STD', linestyle='dashed')
            ax1c.set_ylabel('Steering Wheel STD (degrees)', color='tab:purple', fontsize=8)
            ax1c.tick_params(axis='y', labelcolor='tab:purple')
            ax1.set_title('Lane Deviation STD, Perclos, and Steering Wheel STD Over Time')

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
                    ecg_signal = biopac_df['Biopac_2']

                    # Apply bandpass filter to ECG signal
                    lowcut = 6
                    highcut = 35
                    fs = 500
                    filtered_ecg = bandpass_filter(ecg_signal, lowcut, highcut, fs)
                    # Process the ECG signal to detect R peaks
                    processed_ecg = nk.ecg_process(filtered_ecg, sampling_rate=fs)
                    r_peaks = processed_ecg[1]['ECG_R_Peaks']
                    rr_intervals = np.diff(r_peaks) * (1 / fs)
                    heart_rate = 60 / rr_intervals
                    hr_times = biopac_df.index[r_peaks[1:]]
                    hr_series = pd.Series(heart_rate, index=hr_times)

                    # Normalize the filtered ECG signal between 0 and 1
                    normalized_ecg = (filtered_ecg - np.min(filtered_ecg)) / (np.max(filtered_ecg) - np.min(filtered_ecg))

                    # Plotting the heart rate
                    ax2.plot(hr_series.index / 60, hr_series, color='tab:blue', label='Heart Rate (BPM)')
                    ax2.set_ylabel('Heart Rate (BPM)', color='tab:blue', fontsize=8)
                    ax2.set_xlabel('Time', fontsize=8)
                    ax2.tick_params(axis='y', labelcolor='tab:blue')
                    ax2.legend(loc='upper right', fontsize=8)
                    ax2.set_title('Heart Rate')

                    # Create a secondary y-axis for the normalized ECG signal
                    #ax2b = ax2.twinx()
                    #ax2b.plot(biopac_df.index / 60, normalized_ecg, color='tab:red', label='Filtered ECG Signal (Normalized)')
                    #ax2b.set_ylabel('Filtered ECG Signal (Normalized)', color='tab:red')
                    #ax2b.tick_params(axis='y', labelcolor='tab:red')
                    #ax2b.legend(loc='upper left')

                    # Add markers at R peak indices
                    #r_peak_times = biopac_df.index[r_peaks] / 60  # Convert to minutes
                    #ax2b.plot(r_peak_times, normalized_ecg[r_peaks], 'o', color='tab:green', label='R Peaks')

                    # Adding legends and showing the plot
                    #ax2b.legend(loc='upper left')
        

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

                ax3.set_ylabel('Road position (m)', fontsize=8)
                ax3.set_xlabel('Time', fontsize=8)
                ax3.legend(loc='upper right')
                ax3.set_title('Road position over time')

            # Plotting Steering Wheel Compensated
            steering_wheel_compensated = simulator_df['Steering_Wheel_Compensated']
            ax4.plot(steering_wheel_compensated.index/60, steering_wheel_compensated, color='tab:purple')
            ax4.set_ylabel('Steering position (degrees)', fontsize=8)
            ax4.set_xlabel('Time', fontsize=8)
            ax4.set_title('Steering Wheel Compensated Over Time')
            plt.tight_layout(pad=6.0)
            plt.subplots_adjust(hspace=0.2)
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

# Helper function for bandpass filtering
def bandpass_filter(data, lowcut, highcut, fs, order=1):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data)
    return y
# Helper function for adaptive thresholding
def adaptive_thresholding(ecg_signal, initial_r_peaks, fs, window_size=0.25):
    # Convert window size to samples
    window_size_samples = int(window_size * fs)
    
    refined_r_peaks = []
    for i in range(len(initial_r_peaks)):
        # Define the start and end of the segment around the initial peak
        start = max(0, initial_r_peaks[i] - window_size_samples // 2)
        end = min(len(ecg_signal), initial_r_peaks[i] + window_size_samples // 2)
        
        # Extract the segment of the signal
        segment = ecg_signal[start:end]
        
        # Calculate the adaptive threshold for the segment
        threshold = np.mean(segment) + 3* np.std(segment)
        
        # Find local maxima within the segment that exceed the threshold
        peaks, _ = find_peaks(segment, height=threshold)
        
        if len(peaks) > 0:
            # Select the peak closest to the center of the segment
            peak_distances = np.abs(peaks - (window_size_samples // 2))
            r_peak_candidate = peaks[np.argmin(peak_distances)] + start
            refined_r_peaks.append(r_peak_candidate)
    
    return np.array(refined_r_peaks)

def verify_and_remove_duplicate_peaks(peaks):
    differences = np.diff(peaks)
    duplicate_indices = np.where(differences == 0)[0]
    if len(duplicate_indices) > 0:
        print(f"Duplicate peaks found at indices: {duplicate_indices}")
    peaks = np.delete(peaks, duplicate_indices + 1)
    return peaks

def create_common_time_base(biopac_hr_series, stm32_hr_series, freq=1):
    """Create a common time base for resampling."""
    start_time = max(biopac_hr_series.index[0], stm32_hr_series.index[0])
    end_time = min(biopac_hr_series.index[-1], stm32_hr_series.index[-1])
    common_time_base = np.arange(start_time, end_time + freq, freq)
    return common_time_base

def resample_bpm(bpm_series, target_time_index):
    """Resample BPM series to match the target time index."""
    original_time_index = bpm_series.index
    bpm_values = bpm_series.values

    # Create an interpolation function
    interp_func = interp1d(original_time_index, bpm_values, kind='linear', fill_value='extrapolate')

    # Resample BPM values to match the target time index
    resampled_bpm_values = interp_func(target_time_index)
    resampled_bpm_series = pd.Series(resampled_bpm_values, index=target_time_index)
    
    return resampled_bpm_series

def advanced_bpm_correction(resampled_biopac_hr_series, resampled_stm32_hr_series, avg_window_size=60):
    # Initialize the corrected Biopac BPM series
    corrected_biopac_hr_series = resampled_biopac_hr_series.copy()

    # Convert to arrays for easier manipulation
    biopac_bpm = resampled_biopac_hr_series.values
    stm32_bpm = resampled_stm32_hr_series.values

    last_correct_biopac_bpm = None
    last_correct_stm32_bpm = None

    for i in range(len(biopac_bpm)):
        # Check if the new BPM of Biopac is bad
        is_biopac_bad = (
            biopac_bpm[i] < 40 or biopac_bpm[i] > 150 or
            (i > 0 and abs(biopac_bpm[i] - biopac_bpm[i - 1]) > 15)
        )

        # Check if the new BPM of STM32 is bad
        is_stm32_bad = (
            stm32_bpm[i] < 40 or stm32_bpm[i] > 150 or
            (i > 0 and abs(stm32_bpm[i] - stm32_bpm[i - 1]) > 15)
        )

        if is_biopac_bad:
            if not is_stm32_bad:
                # Replace Biopac BPM with STM32 BPM if STM32 BPM is good
                corrected_biopac_hr_series.iloc[i] = stm32_bpm[i]
                last_correct_biopac_bpm = stm32_bpm[i]
            else:
                # Replace Biopac BPM with last correct value if STM32 BPM is also bad
                corrected_biopac_hr_series.iloc[i] = last_correct_biopac_bpm
        
        if not is_stm32_bad:
            last_correct_stm32_bpm = stm32_bpm[i]

        # If Biopac BPM is still bad after replacement, use average of the last 60 values
        if is_biopac_bad:
            if i >= avg_window_size:
                avg_last_60_biopac = np.mean(corrected_biopac_hr_series[i - avg_window_size:i])
            else:
                avg_last_60_biopac = np.mean(corrected_biopac_hr_series[:i])
            corrected_biopac_hr_series.iloc[i] = avg_last_60_biopac

    return corrected_biopac_hr_series

def plot_ecg_signals_V_1_0(resampled_dfs, patient, session, plotting_activated):
    plotting = int(plotting_activated)
    full_session = f"{patient}_{session}"
    adaptive_thresholding_activated=1
    Map_quality_ECG_path = r'E:\Recordings\Qualité_signal_ECG.csv'
    if os.path.exists(Map_quality_ECG_path):
        Map_quality_ECG_df = pd.read_csv(Map_quality_ECG_path)
    patient_session = patient + "_" + session
    print(patient_session)

    ECG_signal_quality = Map_quality_ECG_df[Map_quality_ECG_df['signal'] == patient_session]
    print(ECG_signal_quality)

    try:
        biopac_df = resampled_dfs[f'{full_session}_Biopac.csv']
        STM32_ECG = resampled_dfs[f'{full_session}_Stm32ECG.csv']
        
        if 'Biopac_2' in biopac_df.columns:
            ecg_signal = biopac_df['Biopac_2']

            # Apply bandpass filter to ECG signal
            lowcut = 6
            highcut = 35
            fs = 500
            filtered_ecg = bandpass_filter(ecg_signal, lowcut, highcut, fs,5)
            normalized_ecg = (filtered_ecg - np.min(filtered_ecg)) / (np.max(filtered_ecg) - np.min(filtered_ecg))
            normalized_ecg = normalized_ecg ** 2

            # Process the ECG signal to detect initial R peaks
            processed_ecg = nk.ecg_process(filtered_ecg, sampling_rate=fs)
            r_Peaks_biopac = processed_ecg[1]['ECG_R_Peaks']
            # Apply adaptive thresholding to refine R peaks
            if adaptive_thresholding_activated==1:
                refined_r_peaks_biopac = adaptive_thresholding(filtered_ecg, r_Peaks_biopac, fs)
                r_Peaks_biopac = verify_and_remove_duplicate_peaks(refined_r_peaks_biopac)
            # Calculate heart rate
            heart_rate = 60 / np.diff(r_Peaks_biopac) * fs
            hr_times = biopac_df.index[r_Peaks_biopac[1:]]
            hr_series = pd.Series(heart_rate, index=hr_times)
            print('Biopac_2')
            # Initialize lists for STM32 HR data
            hr_series_list = []
            selected_STM_ECG = []
            peak_indices_dict = {}  # Dictionary to store peak indices for STM32 signals

            # Process STM32 ECG signals and calculate heart rate
            for column in ['Stm32ECG_0', 'Stm32ECG_1', 'Stm32ECG_2']:
                if column in STM32_ECG.columns:
                    if int(ECG_signal_quality[column]) == 1:
                        ecg_signal_stm = STM32_ECG[column]
                        filtered_ecg_stm = bandpass_filter(ecg_signal_stm, lowcut, highcut, fs,5)
                        filtered_ecg_stm = filtered_ecg_stm ** 2
                        processed_ecg_stm = nk.ecg_process(filtered_ecg_stm, sampling_rate=fs)
                        r_peaks_stm = processed_ecg_stm[1]['ECG_R_Peaks']

                        if adaptive_thresholding_activated==1:
                            refined_r_peaks_stm = adaptive_thresholding(filtered_ecg_stm, r_peaks_stm, fs)
                            r_peaks_stm = verify_and_remove_duplicate_peaks(refined_r_peaks_stm)

                        heart_rate_stm = 60 / np.diff(r_peaks_stm) * fs
                        hr_times_stm = STM32_ECG.index[r_peaks_stm[1:]]
                        hr_series_stm = pd.Series(heart_rate_stm, index=hr_times_stm)
                        hr_series_list.append((hr_series_stm, column))
                        print(column)
                        selected_STM_ECG.append(column)
                        peak_indices_dict[column] = r_peaks_stm  # Store peak indices
            
            # Resample the HR series to a common time base for plotting
            if hr_series_list:
                stm32_hr_series = hr_series_list[0][0]  # Use the first STM32 HR series for correction
                common_time_base = create_common_time_base(hr_series, stm32_hr_series)
                resampled_biopac_hr = resample_bpm(hr_series, common_time_base)
                resampled_stm32_hr = resample_bpm(stm32_hr_series, common_time_base)
                corrected_hr_series = advanced_bpm_correction(resampled_biopac_hr, resampled_stm32_hr)
                
                if plotting == 1:
                    # Create subplots with shared x-axis
                    fig, ax = plt.subplots(3, 1, figsize=(16, 30), sharex=True)
                    # Plotting the heart rate for biopac
                    ax[0].plot(biopac_df.index, normalized_ecg, color='tab:red', label='Biopac Filtered ECG Signal (Normalized)')
                    ax[0].set_ylabel('Filtered ECG Signal (Normalized)', color='tab:red')
                    ax[0].set_xlabel('Time (s)')
                    ax[0].tick_params(axis='y', labelcolor='tab:red')
                    ax[0].legend(loc='upper left')
                    # Add markers at R peak indices
                    r_peak_times = biopac_df.index[r_Peaks_biopac]
                    ax[0].plot(r_peak_times, normalized_ecg[r_Peaks_biopac], 'o', color='tab:green', label='R Peaks')
                    # Adding legends and showing the plot
                    ax[0].legend(loc='upper left')

                    # Plot STM32 ECG signals
                    for index, column in enumerate(selected_STM_ECG):
                        if column in STM32_ECG.columns:
                            STMECG_Filtered = bandpass_filter(STM32_ECG[column], 4, 35, 500, 5)
                            normalized_STMECG_Filtered = (STMECG_Filtered - np.min(STMECG_Filtered)) / (np.max(STMECG_Filtered) - np.min(STMECG_Filtered)) 
                            ax[1].plot(STM32_ECG.index, normalized_STMECG_Filtered + index, label=column)
                            # Add markers at R peak indices for STM32 signals
                            r_peak_times_stm = STM32_ECG.index[peak_indices_dict[column]]
                            ax[1].plot(r_peak_times_stm, normalized_STMECG_Filtered[peak_indices_dict[column]] + index, 'o', color='tab:green', label=f'{column} R Peaks')

                    ax[1].set_ylabel('ECG Signals')
                    ax[1].set_xlabel('Time (s)')
                    ax[1].legend(loc='upper right')
                    ax[1].set_title('STM32 ECG Signals Over Time')

                    # Plotting the heart rate for biopac and STM32 signals
                    ax[2].plot(resampled_biopac_hr.index, resampled_biopac_hr, label='Biopac Heart Rate (BPM)')
                    ax[2].plot(corrected_hr_series.index, corrected_hr_series, label='Biopac Heart Rate corrected', linestyle='--')
                    ax[2].plot(resampled_stm32_hr.index, resampled_stm32_hr, label='STM32 Heart Rate (BPM)')
                    
                    ax[2].set_ylabel('Heart Rate (BPM)', color='tab:blue')
                    ax[2].set_xlabel('Time (s)')
                    ax[2].tick_params(axis='y', labelcolor='tab:blue')
                    ax[2].legend(loc='upper right')
                    ax[2].set_title('Heart Rate from All Signals Over Time')

                    plt.tight_layout(pad=8.0)
                    plt.subplots_adjust(hspace=0.5)
                    plt.show()

    except KeyError as e:
        print(f"KeyError: {e}")
    except Exception as e:
        print(f"General error: {str(e)}")


def plot_ecg_signals_V_2_0(directory, resampled_dfs, patient, session, plotting_activated,Drowsiness_Accuracy_signal_quality_csv_total,Drowsiness_Accuracy_signal_quality_csv_granularized):

    def process_signal_quality_indexes(
        Signal_quality_indexes_midpoints, 
        Signal_quality_indexes_Peaks, 
        patient, 
        session, 
        df_Drowsiness_Accuracy_signal_quality_csv_granularized, 
        ECG_peaks_reference, 
        ECG_peaks_analysed, 
        column, 
        analysed_ECG,
        tolerance=15
    ):
        F1s = []
        # Calculate the number of samples per 10 minutes
        samples_per_10_min = 10 * 60 * 500  # 10 minutes * 60 seconds * 500 Hz

        # Split the analysed_ECG into 10-minute windows
        num_windows = len(analysed_ECG) // samples_per_10_min

        # Convert peaks to numpy arrays for efficient indexing
        ECG_peaks_reference = np.array(ECG_peaks_reference)
        ECG_peaks_analysed = np.array(ECG_peaks_analysed)

        for i in range(num_windows):
            start_index = i * samples_per_10_min
            end_index = (i + 1) * samples_per_10_min
            window = analysed_ECG[start_index:end_index]

            # Filter peaks for the current window
            ref_peaks_in_window = ECG_peaks_reference[(ECG_peaks_reference >= start_index) & (ECG_peaks_reference < end_index)] - start_index
            analysed_peaks_in_window = ECG_peaks_analysed[(ECG_peaks_analysed >= start_index) & (ECG_peaks_analysed < end_index)] - start_index

            # Calculate performance metrics for the current window
            accuracy, f1, recall, specificity = derive_performance_metrics(ref_peaks_in_window, analysed_peaks_in_window, tolerance)
            F1s.append(f1)
            print(f"Window {i}: Accuracy={accuracy}, F1 Score={f1}, Sensitivity={recall}, Specificity={specificity}")

            # Examine each signal quality index for midpoints and ECG peaks in the current window
            for idx, (index_values, midpoints) in enumerate(Signal_quality_indexes_midpoints):
                # Filter midpoints for the current window
                midpoints_in_window = midpoints[(midpoints >= start_index) & (midpoints < end_index)] - start_index
                index_values_in_window = index_values[(midpoints >= start_index) & (midpoints < end_index)]

                # Process or analyze these values as needed
                # For now, we just print them
                print(f"Window {i}, Midpoint Index {idx}: Values={index_values_in_window}, Midpoints={midpoints_in_window}")

            for idx, (index_values, peaks) in enumerate(Signal_quality_indexes_Peaks):
                # Filter peaks for the current window
                peaks_in_window = peaks[(peaks >= start_index) & (peaks < end_index)] - start_index
                index_values_in_window = index_values[(peaks >= start_index) & (peaks < end_index)]
                print(len(ref_peaks_in_window))
                print(len(analysed_peaks_in_window))
                # Process or analyze these values as needed
                # For now, we just print them
                print(f"Window {i}, Peak Index {idx}: Values={index_values_in_window}, Peaks={peaks_in_window}")

        remaining_samples = len(analysed_ECG) % samples_per_10_min
        if remaining_samples >= 5 * 60 * 500:  # 5 minutes * 60 seconds * 500 Hz
            start_index = len(analysed_ECG) - remaining_samples
            window = analysed_ECG[start_index:]

            # Filter peaks for the last window
            ref_peaks_in_window = ECG_peaks_reference[(ECG_peaks_reference >= start_index)] - start_index
            
            analysed_peaks_in_window = ECG_peaks_analysed[(ECG_peaks_analysed >= start_index)] - start_index
            
            # Calculate performance metrics for the last window
            accuracy, f1, recall, specificity = derive_performance_metrics(ref_peaks_in_window, analysed_peaks_in_window, tolerance)
            F1s.append(f1)
            print(f"Last Window: Accuracy={accuracy}, F1 Score={f1}, Sensitivity={recall}, Specificity={specificity}")

            # Examine each signal quality index for midpoints and ECG peaks in the last window
            for idx, (index_values, midpoints) in enumerate(Signal_quality_indexes_midpoints):
                # Filter midpoints for the current window
                midpoints_in_window = midpoints[(midpoints >= start_index)] - start_index
                index_values_in_window = index_values[(midpoints >= start_index)]

                # Process or analyze these values as needed
                # For now, we just print them
                print(f"Last Window, Midpoint Index {idx}: Values={index_values_in_window}, Midpoints={midpoints_in_window}")

            for idx, (index_values, peaks) in enumerate(Signal_quality_indexes_Peaks):
                # Filter peaks for the current window
                peaks_in_window = peaks[(peaks >= start_index)] - start_index
                index_values_in_window = index_values[(peaks >= start_index)]

                # Process or analyze these values as needed
                # For now, we just print them
                print(f"Last Window, Peak Index {idx}: Values={index_values_in_window}, Peaks={peaks_in_window}")

        return F1s
            
        
 
    def extract_beats(signal, peaks, window=100):
        beats = []
        for peak in peaks:
            start = max(peak - window, 0)
            end = min(peak + window, len(signal) - 1)
            beat = signal[start:end+1]
            if len(beat) == 2*window + 1:
                beat_with_indices = np.array([[i, beat[i]] for i in range(len(beat))])
                beats.append(beat_with_indices)
        return beats

    def calculate_dtw_distances(signal, peaks, window=100):
        beats = extract_beats(signal, peaks, window)
        
        dtw_distances = []
        
        for i in tqdm(range(len(beats)),desc="Calculating DTW"):
            current_beat_with_indices = beats[i]
            
            if i == 0:
                next_beat_with_indices = beats[i + 1]
                distance, _ = fastdtw(current_beat_with_indices, next_beat_with_indices, dist=euclidean)
            elif i == len(beats) - 1:
                previous_beat_with_indices = beats[i - 1]
                distance, _ = fastdtw(current_beat_with_indices, previous_beat_with_indices, dist=euclidean)
            else:
                previous_beat_with_indices = beats[i - 1]
                next_beat_with_indices = beats[i + 1]
                
                #distance_prev, _ = fastdtw(current_beat_with_indices, previous_beat_with_indices, dist=euclidean)
                distance_next, _ = fastdtw(current_beat_with_indices, next_beat_with_indices, dist=euclidean)
                
                #distance = (distance_prev + distance_next) / 2
                distance =distance_next
            
            dtw_distances.append(distance)
            
        return np.array(dtw_distances)

    
    def moving_window_mad(signal, window_size, sampling_rate):
        window_samples = int(window_size * sampling_rate)
        mad_values = []
        for i in tqdm(range(0, len(signal) - window_samples + 1, sampling_rate), desc="Calculating MAD"):
            window = signal[i:i+window_samples]
            mad_value = np.mean(np.abs(window - np.mean(window)))
            mad_values.append(mad_value)
        return np.array(mad_values)

    def moving_window_std_amplitude_peaks_to_peaks(ecg_signal, peak_indices, window_size=5):
        std_amplitude_peaks_to_peaks = [0] * (window_size - 1)  # Initialize with zeros for the first (window_size - 1) values
        for i in tqdm(range(len(peak_indices) - window_size + 1), desc="Calculating STD Amplitude Peaks-to-Peaks"):
            window_peaks = ecg_signal[peak_indices[i:i + window_size]]
            std_amplitude_peaks_to_peaks.append(np.std(window_peaks))
        return std_amplitude_peaks_to_peaks

    def calculate_std_ratio(signal, peaks,column, inner_window=25, outer_window=100):
        ratios = []
        for peak in peaks:
            start_inner = max(peak - inner_window, 0)
            end_inner = min(peak + inner_window, len(signal) - 1)
            start_outer = max(peak - outer_window, 0)
            end_outer = min(peak + outer_window, len(signal) - 1)
            
            inner_segment = signal[start_inner:end_inner+1]
            outer_segment = signal[start_outer:end_outer+1]
            
            std_inner = np.std(inner_segment)
            std_outer = np.std(outer_segment)
            
            ratio = std_inner / (2 * std_outer) if std_outer != 0 else np.nan
            ratios.append(ratio)
        return np.array(ratios)

    def normalize_signal(signal):
        return (signal - np.min(signal)) / (np.max(signal) - np.min(signal))

    def calculate_performance_metrics(true_peaks, detected_peaks, tolerance):
        true_peaks = np.array(true_peaks)
        detected_peaks = np.array(detected_peaks)

        tp = 0
        fn = 0
        fp = 0

        for true_peak in true_peaks:
            if np.any(np.abs(detected_peaks - true_peak) <= tolerance):
                tp += 1
            else:
                fn += 1

        for detected_peak in detected_peaks:
            if not np.any(np.abs(true_peaks - detected_peak) <= tolerance):
                fp += 1

        return tp, fn, fp
    def derive_performance_metrics(ECG_peaks_reference,ECG_peaks_analysed,tolerance):
        tp, fn, fp = calculate_performance_metrics(ECG_peaks_reference,ECG_peaks_analysed, tolerance)
        tn = len(ECG_peaks_reference) - tp - fn  # true negatives not directly computable here

        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp) if tp + fp > 0 else 0
        recall = tp / (tp + fn) if tp + fn > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
        specificity = tn / (tn + fp) if tn + fp > 0 else 0
        return accuracy,f1,recall,specificity
    
    def save_dictionary(filepath, data_dict):
        with open(filepath, 'wb') as file:
            pickle.dump(data_dict, file)
        print(f"Saved dictionary data to {filepath}")

    def load_dictionary(filepath):
        with open(filepath, 'rb') as file:
            return pickle.load(file)
    def moving_window_kurtosis(signal, window_size, sampling_rate,column):
        window_samples = int(window_size * sampling_rate)
        kurtosis_values = []
        for i in tqdm(range(0, len(signal) - window_samples + 1, sampling_rate), desc="Calculating Kurtosis"):
            window = signal[i:i+window_samples]
            if np.std(window, ddof=1) != 0:  # Check to prevent division by zero
                standardized_window = (window - np.mean(window)) / np.std(window, ddof=1)
                kurtosis_value = kurtosis(standardized_window)
                if (column!='Biopac_ECG') and kurtosis_value >=20:
                    kurtosis_values.append(0)
                else:
                    kurtosis_values.append(kurtosis_value)
            else:
                kurtosis_values.append(np.nan)
        return np.array(kurtosis_values)

    def moving_window_skewness(signal, window_size,sampling_rate,column):
        window_samples = int(window_size * sampling_rate)
        skewness_values = []
        for i in tqdm(range(0, len(signal) - window_samples + 1, sampling_rate), desc="Calculating Skewness"):
            window = signal[i:i+window_samples]
            if np.std(window, ddof=1) != 0:  # Check to prevent division by zero
                standardized_window = (window - np.mean(window)) / np.std(window, ddof=1)
                skewness_value = skew(standardized_window)
                skewness_values.append(skewness_value)
            else:
                skewness_values.append(np.nan)
        return np.array(skewness_values)
    
    def compute_midpoints(ecg_time, window_size, sampling_rate):
        window_samples = int(window_size * sampling_rate)
        midpoints = [ecg_time[i + window_samples // 2] for i in range(0, len(ecg_time) - window_samples + 1, sampling_rate)]
        return midpoints

    def moving_window_spectral_power_ratio(signal, window_size, sampling_rate, column):
        # Design the notch filter
        notch_freq = 40.0  # Frequency to be removed from signal (Hz)
        quality_factor = 30.0  # Quality factor for notch filter
        b, a = iirnotch(notch_freq, quality_factor, sampling_rate)
        
        # Apply the notch filter to the entire signal
        filtered_signal = filtfilt(b, a, signal)
        
        window_samples = int(window_size * sampling_rate)
        power_ratios_5_15 = []
        power_ratios_5_45 = []
        power_ratios_0_5 = []
        
        for i in tqdm(range(0, len(filtered_signal) - window_samples + 1, sampling_rate), desc="Calculating Spectral Power Ratios"):
            window = filtered_signal[i:i + window_samples]
            freqs, psd = welch(window, fs=sampling_rate)
            
            band_5_15 = np.logical_and(freqs >= 5, freqs <= 15)
            band_5_45 = np.logical_and(freqs >= 5, freqs <= 45)
            band_0_5 = np.logical_and(freqs >= 0, freqs <= 5)
            band_0_45 = np.logical_and(freqs >= 0, freqs <= 45)
            
            power_5_15 = np.sum(psd[band_5_15])
            power_5_45 = np.sum(psd[band_5_45])
            power_0_5 = np.sum(psd[band_0_5])
            power_0_45 = np.sum(psd[band_0_45])
            
            power_ratio_5_15 = power_5_15 / power_0_45 if power_0_45 > 0 else np.nan
            power_5_45 = power_5_45 / power_0_45 if power_0_45 > 0 else np.nan
            power_ratio_0_5 = power_0_5 / power_0_45 if power_0_45 > 0 else np.nan
            
            power_ratios_5_15.append(power_ratio_5_15)
            power_ratios_5_45.append(power_5_45)
            power_ratios_0_5.append(power_ratio_0_5)
        
        return np.array(power_ratios_5_15), np.array(power_ratios_5_45), np.array(power_ratios_0_5)

    
    def calculate_power(signal, index, window_size,column):
        half_window = window_size // 2
        start = max(index - half_window, 0)
        end = min(index + half_window + 1, len(signal))
        window = signal[start:end]
        power = np.sum(window**2)
        return power

    def calculate_power_ratios(signal, peaks, narrow_window, wide_window,column):
        narrow_powers = [calculate_power(signal, peak, narrow_window,column=column) for peak in peaks]
        wide_powers = [calculate_power(signal, peak, wide_window,column=column) for peak in peaks]
        power_ratios = np.array(narrow_powers) / np.array(wide_powers)
        return power_ratios

    lowcut = 4
    highcut = 35
    fs = 500
    sampling_rate = 500 
    ECGs_filtered = {}
    ECGs_non_filtered = {}
    ECG_peaks = {}
    plotting = int(plotting_activated)
    full_session = f"{patient}_{session}"


    biopac_df = resampled_dfs.get(f'{full_session}_Biopac.csv')
    if biopac_df is None:
        print(f"Biopac data for {full_session} not found.")
        return
            
    STM32_ECG = resampled_dfs.get(f'{full_session}_Stm32ECG.csv')
    if STM32_ECG is None:
        print(f"STM32 ECG data for {full_session} not found.")
        return
        
    # Process ECG signal using neurokit2's ecg_process
    ECG_identifiants = ['Biopac_ECG','Stm32ECG_0', 'Stm32ECG_1', 'Stm32ECG_2']
    map_quality_ecg_path = r'E:\Recordings\Qualité_signal_ECG.csv'
    
    if os.path.exists(map_quality_ecg_path):
        map_quality_ecg_df = pd.read_csv(map_quality_ecg_path)
    else:
        print(f"File {map_quality_ecg_path} does not exist.")
        return
    
    patient_session = f"{patient}_{session}"
    print(patient_session)

    ecg_signal_quality = map_quality_ecg_df[map_quality_ecg_df['signal'] == patient_session]
    print(ecg_signal_quality)

    Pre_filtered_ECG_file_path = os.path.join(directory,patient,full_session, f'Prefiltered_ECG.csv')
    Precomputed_peaks_file_path = os.path.join(directory,patient,full_session, f'Precomputed_peaks.csv')
    Pre_non_filtered_ECG_file_path = os.path.join(directory, patient, full_session, 'Prenonfiltered_ECG.csv')

    if os.path.exists(Pre_filtered_ECG_file_path) and os.path.exists(Precomputed_peaks_file_path) and os.path.exists(Pre_non_filtered_ECG_file_path):
        ECGs_filtered = load_dictionary(Pre_filtered_ECG_file_path)
        print(ECGs_filtered)
        ECG_peaks = load_dictionary(Precomputed_peaks_file_path)
        print(ECG_peaks)
        ECGs_non_filtered = load_dictionary(Pre_non_filtered_ECG_file_path)
        print(ECGs_non_filtered)
        print(f"Loaded Pre_filtered_ECG data from {Pre_filtered_ECG_file_path}")
        print(f"Loaded Precomputed ECG peaks data from {Precomputed_peaks_file_path}")
        print(f"Loaded Prenonfiltered_ECG data from {Pre_non_filtered_ECG_file_path}")
    else:
        

        for index,column in enumerate(ECG_identifiants):
                if index==0:
                    analysed_ECG = biopac_df['Biopac_2']
                    analysed_ECG=normalize_signal(analysed_ECG)
                else:
                    analysed_ECG = STM32_ECG[column]
                    analysed_ECG=normalize_signal(analysed_ECG)
                ECGs_non_filtered[column] = analysed_ECG.to_numpy()

                filtered_signal = bandpass_filter(analysed_ECG, lowcut, highcut, fs, 5)
                ECGs_filtered[column]=filtered_signal
                ecg_signals, ecg_info = nk.ecg_process(filtered_signal, sampling_rate=sampling_rate)
                ECG_peaks[column] = ecg_info['ECG_R_Peaks']
                print(column + "...processed")
        
        # Save the processed signals
        
        # Convert ECGs_filtered to a DataFrame
        save_dictionary(Precomputed_peaks_file_path, ECG_peaks)
        save_dictionary(Pre_filtered_ECG_file_path, ECGs_filtered)
        save_dictionary(Pre_non_filtered_ECG_file_path, ECGs_non_filtered)

    try:
        rows_to_be_added=[]
        # Calculate the time indices for ECG signal
        ecg_time =biopac_df['Biopac_2'].index
        # Plot original ECG signal and heart rate
        fig, ax1 = plt.subplots(4, 1, figsize=(12, 8), sharex=True)
        # Plot skewness in a new figure
        fig2, ax2 = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
        # Plot spectral power ratios in another figure
        fig3, ax3 = plt.subplots(4, 1, figsize=(12, 8), sharex=True)
        # Plotting configurations
        fig4, ax4 = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
            # After the existing plotting code, add a new figure for the STD ratio metric
        fig5, ax5 = plt.subplots(2,1,figsize=(12, 5))   
        
            # After the existing plotting code, add a new figure for the STD ratio metric
        fig6, ax6 = plt.subplots(2,1,figsize=(12, 5))      
        
        # After the existing plotting code, add a new figure for the STD ratio metric
        fig7, ax7 = plt.subplots(2,1,figsize=(12, 5))   
        
        # Add a new figure for the DTW distances
        fig8, ax8 = plt.subplots(2,1,figsize=(12, 5))
        tolerance = 15
        metrics = {}
        Signal_quality_indexes_midpoints=[]
        Signal_quality_indexes_Peaks=[]
        F1s=[]
        Drowsiness_Accuracy_signal_quality_csv_total,Drowsiness_Accuracy_signal_quality_csv_granularized
        lock_file_total = f"{Drowsiness_Accuracy_signal_quality_csv_total}.lock"
        lock = FileLock(lock_file_total, timeout=120)  # Increase timeout if needed
        
        lock_file_Granularized = f"{Drowsiness_Accuracy_signal_quality_csv_granularized}.lock"
        
        try:
            with lock:
                if os.path.exists(Drowsiness_Accuracy_signal_quality_csv_total):
                    print(f"{Drowsiness_Accuracy_signal_quality_csv_total} file exists")
                    df_Drowsiness_Accuracy_signal_quality_csv_total = pd.read_csv(Drowsiness_Accuracy_signal_quality_csv_total)  # Read the existing file
                else:
                    print(f"{Drowsiness_Accuracy_signal_quality_csv_total} doesn't exist. Creating the file...")
                    # Define the headers based on the data you want to store
                    headers = ['Patient', 'Session','STM32_ECG_signal', 'Accuracy', 'F1 Score', 'Sensitivity', 'Specificity','Average_skewness','Average_kurtosis','Average_power_ratios_5_15','Average_power_ratios_5_45','Average_power_ratios_0_5','Average_power_ratios_peaks','Average_std_ratios_peaks','STD_amplitude_Peaks_to_peaks','dtw_avg_distances_between_peaks']
                    # Create an empty DataFrame with the defined headers
                    df_Drowsiness_Accuracy_signal_quality_csv_total = pd.DataFrame(columns=headers)
                    
                    # Save the empty DataFrame to a CSV file
                    df_Drowsiness_Accuracy_signal_quality_csv_total.to_csv(Drowsiness_Accuracy_signal_quality_csv_total, index=False)
                    print(f"{Drowsiness_Accuracy_signal_quality_csv_total} file created successfully.")
                    
                if os.path.exists(Drowsiness_Accuracy_signal_quality_csv_granularized):
                    print(f"{Drowsiness_Accuracy_signal_quality_csv_granularized} file exists")
                    df_Drowsiness_Accuracy_signal_quality_csv_granularized = pd.read_csv(Drowsiness_Accuracy_signal_quality_csv_granularized)  # Read the existing file
                else:
                    print(f"{Drowsiness_Accuracy_signal_quality_csv_granularized} doesn't exist. Creating the file...")
                    # Define the headers based on the data you want to store
                    headers = ['Patient', 'Session','section','STM32_ECG_signal', 'Accuracy', 'F1 Score', 'Sensitivity', 'Specificity','Average_skewness','Average_kurtosis','Average_power_ratios_5_15','Average_power_ratios_5_45','Average_power_ratios_0_5','Average_power_ratios_peaks','Average_std_ratios_peaks','STD_amplitude_Peaks_to_peaks','dtw_avg_distances_between_peaks']
                    # Create an empty DataFrame with the defined headers
                    df_Drowsiness_Accuracy_signal_quality_csv_granularized = pd.DataFrame(columns=headers)
                    
                    # Save the empty DataFrame to a CSV file
                    df_Drowsiness_Accuracy_signal_quality_csv_granularized.to_csv(Drowsiness_Accuracy_signal_quality_csv_granularized, index=False)
                    print(f"{Drowsiness_Accuracy_signal_quality_csv_granularized} file created successfully.")
        except Timeout:
            print(f"Could not acquire the lock within the timeout period for {lock_file_total}. Exiting process.")
            raise         
        
               
        ##TABLEAU 1 : ECG RAW| ECG PROCESSED| KURTOSIS
        for index,column in enumerate(ECG_identifiants):
                if column in ECG_identifiants:
                    row = {}
                    print(f"analysing {column}:")
                    accuracy,f1,recall,specificity=derive_performance_metrics(ECG_peaks['Biopac_ECG'], ECG_peaks[column], tolerance)
        
                    skewness_values = moving_window_skewness(ECGs_filtered[column],column=column, window_size=10, sampling_rate=sampling_rate)
                    skewness_midpoints = compute_midpoints(ecg_time, 10, sampling_rate)  # For 10-second windows
                    Signal_quality_indexes_midpoints.append((skewness_values,skewness_midpoints))
                    
                    kurtosis_values = moving_window_kurtosis(ECGs_filtered[column],column=column, window_size=10, sampling_rate=sampling_rate)
                    kurtosis_midpoints = compute_midpoints(ecg_time, 10, sampling_rate)  # For 10-second windows
                    Signal_quality_indexes_midpoints.append((kurtosis_values,kurtosis_midpoints))
                    
                    power_ratios_5_15, power_ratios_5_45, power_ratios_0_5 = moving_window_spectral_power_ratio(ECGs_non_filtered[column], window_size=10, sampling_rate=sampling_rate,column=column)
                    midpoints_power_ratios = compute_midpoints(ecg_time, 10, sampling_rate)
                    Signal_quality_indexes_midpoints.append((power_ratios_5_15,midpoints_power_ratios))
                    Signal_quality_indexes_midpoints.append((power_ratios_5_45,midpoints_power_ratios))
                    Signal_quality_indexes_midpoints.append((power_ratios_0_5,midpoints_power_ratios))
                    
                    power_ratios_peaks = calculate_power_ratios(np.clip(ECGs_filtered[column], a_min=0, a_max=None), ECG_peaks[column], 13, 50,column=column)
                    Signal_quality_indexes_Peaks.append((power_ratios_peaks,ECG_peaks[column]))
                    
                    std_ratios_peaks = calculate_std_ratio(ECGs_filtered[column], ECG_peaks[column],column=column)
                    Signal_quality_indexes_Peaks.append((std_ratios_peaks,ECG_peaks[column]))
                    
                    std_amplitude_peaks_to_peaks = moving_window_std_amplitude_peaks_to_peaks(normalize_signal(ECGs_filtered[column]), ECG_peaks[column], window_size=10)
                    Signal_quality_indexes_Peaks.append((std_amplitude_peaks_to_peaks,ECG_peaks[column]))
                    
                    ECG_raw_MAD=moving_window_mad(ECGs_non_filtered[column], window_size=10, sampling_rate=sampling_rate)
                    midpoints_ECG_raw_MAD = compute_midpoints(ecg_time, 10, sampling_rate)
                    Signal_quality_indexes_midpoints.append((ECG_raw_MAD,midpoints_ECG_raw_MAD))
                    
                    
                    # Calculate DTW distances
                    #dtw_avg_distances = calculate_dtw_distances(normalize_signal(ECGs_filtered[column]), ECG_peaks[column], window=100)
                    #Signal_quality_indexes_Peaks.append((dtw_avg_distances,ECG_peaks[column]))
                    
                    if column!='Biopac_ECG':
                        row['Patient'] = patient
                        row['Session'] = session
                        row['STM32_ECG_signal'] = column
                        row['Accuracy'] = accuracy
                        row['F1 Score'] = f1
                        row['Sensitivity'] = recall
                        row['Specificity'] = specificity
                        row['Average_skewness'] = np.mean(skewness_values)
                        row['Average_kurtosis'] = np.mean(kurtosis_values)
                        row['Average_power_ratios_5_15'] = np.mean(power_ratios_5_15)
                        row['Average_power_ratios_5_45'] = np.mean(power_ratios_5_45)
                        row['Average_power_ratios_0_5'] = np.mean(power_ratios_0_5)
                        row['Average_power_ratios_peaks'] =np.nanmean(power_ratios_peaks)
                        row['Average_std_ratios_peaks'] = np.mean(std_ratios_peaks)
                        row['STD_amplitude_Peaks_to_peaks'] = np.mean(std_amplitude_peaks_to_peaks)
                        row['dtw_avg_distances_between_peaks'] = np.mean(std_amplitude_peaks_to_peaks)
                        df_Drowsiness_Accuracy_signal_quality_csv_total = pd.concat([df_Drowsiness_Accuracy_signal_quality_csv_total, pd.DataFrame([row])], ignore_index=True)
                        
                        F1_temp=process_signal_quality_indexes(
                        Signal_quality_indexes_midpoints, 
                        Signal_quality_indexes_Peaks, 
                        patient, 
                        session, 
                        df_Drowsiness_Accuracy_signal_quality_csv_granularized,ECG_peaks['Biopac_ECG'], ECG_peaks[column],column,ECGs_filtered[column]
                        )
                        F1s.append(F1_temp)

                    if plotting ==1:
                        
                        skewness_values_ma = pd.Series(skewness_values).rolling(window=60).mean()
                        kurtosis_values_ma = pd.Series(kurtosis_values).rolling(window=60).mean()
                        power_ratios_5_15_ma = pd.Series(power_ratios_5_15).rolling(window=60).mean()
                        power_ratios_5_45_ma = pd.Series(power_ratios_5_45).rolling(window=60).mean()
                        power_ratios_0_5_ma = pd.Series(power_ratios_0_5).rolling(window=60).mean()
                        power_ratios_peaks_ma = pd.Series(power_ratios_peaks).rolling(window=60).mean()
                        std_ratios_peaks_ma = pd.Series(std_ratios_peaks).rolling(window=60).mean()
                        std_amplitude_peaks_to_peaks_ma = pd.Series(std_amplitude_peaks_to_peaks).rolling(window=60).mean()
                        #dtw_avg_distances_ma=pd.Series(dtw_avg_distances).rolling(window=60).mean()
                        
                        ax1[0].plot(ecg_time, ECGs_filtered[column], label=column)
                        ax1[0].scatter(ecg_time[ECG_peaks[column]], ECGs_filtered[column][ECG_peaks[column]], color='red', label=f'{column} Peaks')

                        ax1[1].plot(ecg_time, ECGs_non_filtered[column], label=column)
                        ax1[1].scatter(ecg_time[ECG_peaks[column]], ECGs_non_filtered[column][ECG_peaks[column]], color='red', label=f'{column} Peaks')
                        ax1[2].plot(kurtosis_midpoints, kurtosis_values_ma, label=f'{column} kurtosis (10s window)')

                        ax2[0].plot(ecg_time, ECGs_filtered[column], label=column)
                        ax2[0].scatter(ecg_time[ECG_peaks[column]], ECGs_filtered[column][ECG_peaks[column]], color='red', label=f'{column} Peaks')
                        ax2[1].plot(skewness_midpoints, skewness_values_ma, label=f'{column} skewness (1s window)')
                        
                        ax3[0].plot(midpoints_power_ratios, power_ratios_5_15_ma, label=f'{column} Power Ratio 5-15 Hz / 0-40 Hz')
                        ax3[1].plot(midpoints_power_ratios, power_ratios_5_45_ma, label=f'{column} Power Ratio 5-40 Hz / 0-40 Hz')
                        ax3[2].plot(midpoints_power_ratios, power_ratios_0_5_ma, label=f'{column} Power Ratio 0-5 Hz / 0-40 Hz')

                        # Plot filtered ECG signals
                        ax4[0].plot(ecg_time, ECGs_filtered[column], label=f'{column} Filtered')
                        ax4[0].scatter(ecg_time[ECG_peaks[column]], ECGs_filtered[column][ECG_peaks[column]], color='red', label='Peaks')
                        ax4[1].plot(ecg_time[ECG_peaks[column]], power_ratios_peaks_ma, label=f'Power Ratio for {column}')

                        ax5[0].plot(ecg_time[ECG_peaks[column]], std_ratios_peaks_ma, label=f'STD Ratio for {column}')
                        
                        ax6[0].plot(ecg_time[ECG_peaks[column]], std_amplitude_peaks_to_peaks_ma, label=f'STD Ratio for {column}')
                        
                        ax7[0].plot(midpoints_ECG_raw_MAD, ECG_raw_MAD, label=f'STD Ratio for {column}')
                        
                        
                        
                        # Plot DTW distances
                        
                        #peaks_length = len(ecg_time[ECG_peaks[column]])
                        #dtw_length = len(dtw_avg_distances)

                        #Adjust lengths to match for plotting
                        #if peaks_length > dtw_length:
                        #    ecg_time_peaks = ecg_time[ECG_peaks[column]][:dtw_length]
                        #else:
                        #    ecg_time_peaks = ecg_time[ECG_peaks[column]]
                        #ax8.plot(ecg_time[ECG_peaks[column]], dtw_avg_distances_ma, label=f'DTW Distances for {column}')
              
        if plotting ==1:  
        
        
            for stm_key in ['Stm32ECG_0', 'Stm32ECG_1', 'Stm32ECG_2']:
                    accuracy,f1,recall,specificity=derive_performance_metrics(ECG_peaks['Biopac_ECG'], ECG_peaks[stm_key], tolerance)
                    metrics[stm_key] = {
                        'Accuracy': accuracy,
                        'F1 Score': f1,
                        'Sensitivity': recall,
                        'Specificity': specificity
                    }

            for key, value in metrics.items():
                print(f"\nMetrics for {key}:")
                for metric, val in value.items():
                    print(f"{metric}: {val:.2f}")
             # Add F1 score plots to each figure
             
            flat_F1s = [f1 for sublist in F1s for f1 in sublist]

            # Define x-axis values for the F1 score plots, spacing by 10 minutes in sample units
            sample_spacing = 10 *60  # 10 minutes * 60 seconds/minute * 500 samples/second
            x_values = [i * sample_spacing for i in range(len(flat_F1s))]

            # Define colors for the plots
            colors = ['orange', 'green', 'red']

            # Plot F1 scores for each column
            for idx, f1_scores in enumerate(F1s):
                num_windows = len(f1_scores)
                window_times = [i * sample_spacing for i in range(num_windows)]  # 10-minute intervals in sample units
                color = colors[idx % len(colors)]  # Cycle through the colors

                ax1[3].plot(window_times, f1_scores, marker='o', linestyle='-', color=color, label=f'F1 Score {idx+1}')
                ax2[2].plot(window_times, f1_scores, marker='o', linestyle='-', color=color, label=f'F1 Score {idx+1}')
                ax3[3].plot(window_times, f1_scores, marker='o', linestyle='-', color=color, label=f'F1 Score {idx+1}')
                ax4[2].plot(window_times, f1_scores, marker='o', linestyle='-', color=color, label=f'F1 Score {idx+1}')
                ax5[1].plot(window_times, f1_scores, marker='o', linestyle='-', color=color, label=f'F1 Score {idx+1}')
                ax6[1].plot(window_times, f1_scores, marker='o', linestyle='-', color=color, label=f'F1 Score {idx+1}')
                ax7[1].plot(window_times, f1_scores, marker='o', linestyle='-', color=color, label=f'F1 Score {idx+1}')
                ax8[1].plot(window_times, f1_scores, marker='o', linestyle='-', color=color, label=f'F1 Score {idx+1}')

            # Set titles and labels for F1 score plots
            for ax in [ax1[3], ax2[2], ax3[3], ax4[2], ax5[1], ax6[1], ax7[1], ax8[1]]:
                ax.set_ylabel('F1 Score')
                ax.legend()
                                           
            ax1[0].set_title('Processed ECG signals')
            ax1[0].set_ylabel('Amplitude')
            ax1[0].set_xlabel('Time')
            ax1[0].legend(loc='upper left')  # Change legend location to bottom left
            ax1[1].set_title('Processed STM32 ECG Signals')
            ax1[1].set_ylabel('Amplitude')
            ax1[1].set_xlabel('Time')
            ax1[1].legend(loc='upper left')  # Change legend location to bottom left
            ax1[2].set_title('Kurtosis  of  ECG Signals')
            ax1[2].set_ylabel('Kurtosis')
            ax1[2].set_xlabel('Time')
            ax1[2].legend(loc='upper left')
            ax1[2].set_ylim(-1, 25)

            ax2[0].set_title('Processed ECG signals')
            ax2[0].set_ylabel('Amplitude')
            ax2[0].set_xlabel('Time')
            ax2[0].legend(loc='upper left')
            ax2[1].set_title('Skewness of ECG Signals')
            ax2[1].set_ylabel('Value')
            ax2[1].set_xlabel('Time')
            ax2[1].legend(loc='upper left')
            

            ax3[0].set_title('Spectral Power Ratios of Biopac ECG Signal and STM32 signals')
            ax3[0].set_ylabel('Power Ratio 5-15 Hz / 0-40 Hz')
            ax3[0].legend(loc='upper left')
            ax3[1].set_ylabel('Power Ratio 5-40 Hz / 0-40 Hz')
            ax3[1].legend(loc='upper left')
            ax3[2].set_ylabel('Power Ratio 0-5 Hz / 0-40 Hz')
            ax3[2].legend(loc='upper left')
            ax3[2].set_xlabel('Time')
                
        
            ax4[0].set_title(f'Filtered ECGs Signals')
            ax4[0].legend(loc='upper left')    
            ax4[1].set_title('Power Ratio (Narrow/Wide) around Peaks')
            ax4[1].set_ylabel('Power Ratio')
            ax4[1].legend(loc='upper left')

            ax5[0].set_title('STD Ratio (Inner/Outer) around Peaks')
            ax5[0].set_ylabel('STD Ratio')
            ax5[0].set_xlabel('Time')
            ax5[0].legend(loc='upper left')

            ax6[0].set_title('STD Amplitude Peaks-to-Peaks (10 beats window)')
            ax6[0].set_xlabel('Time')
            ax6[0].set_ylabel('STD Amplitude Peaks-to-Peaks')
            ax6[0].legend()
            
            ax7[0].set_title('MAD of ECG signals')
            ax7[0].set_xlabel('Time')
            ax7[0].set_ylabel('MAD')
            ax7[0].legend()
            
            ax8[0].set_title('average DTW of ECG signals from one beat to an other')
            ax8[0].set_xlabel('Time')
            ax8[0].set_ylabel('DTW')
            ax8[0].legend()
    
            plt.tight_layout()
            plt.show()
        try:
            with lock:    
                df_Drowsiness_Accuracy_signal_quality_total.to_csv(Drowsiness_Accuracy_signal_quality_csv, index=False)
                print("added row to df_Drowsiness_Accuracy_signal_quality_total correctly")
        except Timeout:
            print(f"Could not acquire the lock within the timeout period for {lock_file_total}. Exiting process.")
            raise 
    except KeyError as e:
        print(f"KeyError: {e}")
    except Exception as e:
        print(f"General error: {str(e)}")
        raise  # This will print the full traceback


def plot_ppg_signal(resampled_dfs, patient, session, plotting_activated):
    plotting = int(plotting_activated)
    full_session = f"{patient}_{session}"

    try:
        biopac_df = resampled_dfs[f'{full_session}_Biopac.csv']
        
        if 'Biopac_1' in biopac_df.columns:
            ppg_signal = biopac_df['Biopac_1']

            # Apply bandpass filter to PPG signal
            lowcut = 0.5
            highcut = 5
            fs = 500
            filtered_ppg = bandpass_filter(ppg_signal, lowcut, highcut, fs,1)
            normalized_ppg = (filtered_ppg - np.min(filtered_ppg)) / (np.max(filtered_ppg) - np.min(filtered_ppg))
            normalized_ppg = normalized_ppg ** 2

            # Process the PPG signal to detect initial peaks
            processed_ppg = nk.ppg_process(filtered_ppg, sampling_rate=fs)
            r_Peaks_ppg = processed_ppg[1]['PPG_Peaks']

            # Calculate heart rate from PPG
            heart_rate = 60 / np.diff(r_Peaks_ppg) * fs
            hr_times = biopac_df.index[r_Peaks_ppg[1:]]
            hr_series = pd.Series(heart_rate, index=hr_times)
            print('Biopac_1')

            if plotting == 1:
                # Create subplots with shared x-axis
                fig, ax = plt.subplots(2, 1, figsize=(16, 20), sharex=True)

                # Plotting the PPG signal
                ax[0].plot(biopac_df.index, normalized_ppg, color='tab:red', label='Biopac Filtered PPG Signal (Normalized)')
                ax[0].set_ylabel('Filtered PPG Signal (Normalized)', color='tab:red')
                ax[0].set_xlabel('Time (s)')
                ax[0].tick_params(axis='y', labelcolor='tab:red')
                ax[0].legend(loc='upper left')

                # Add markers at peak indices
                r_peak_times = biopac_df.index[r_Peaks_ppg]
                ax[0].plot(r_peak_times, normalized_ppg[r_Peaks_ppg], 'o', color='tab:green', label='Peaks')
                ax[0].legend(loc='upper left')

                # Plotting the heart rate derived from PPG
                ax[1].plot(hr_series.index, hr_series, label='PPG Heart Rate (BPM)')
                ax[1].set_ylabel('Heart Rate (BPM)', color='tab:blue')
                ax[1].set_xlabel('Time (s)')
                ax[1].tick_params(axis='y', labelcolor='tab:blue')
                ax[1].legend(loc='upper right')
                ax[1].set_title('Heart Rate from PPG Signal Over Time')

                plt.tight_layout(pad=8.0)
                plt.subplots_adjust(hspace=0.5)
                plt.show()

    except KeyError as e:
        print(f"KeyError: {e}")
    except Exception as e:
        print(f"General error: {str(e)}")