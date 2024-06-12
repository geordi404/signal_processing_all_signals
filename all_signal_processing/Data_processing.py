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
    identifiant_route_path = r'F:\Recordings\Identifiant_route.csv'
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

        #number_of_crash = plot_data_PERCLOS_LANE_DEVIATION(resampled_dfs, patient, session, Perclos_window_size, Lane_deviation_window_size, steering_wheel_std, plotting_activated)
        plot_ecg_signals(resampled_dfs, patient, session, plotting_activated)
    else:
        print("No dataframes were loaded, check file paths and file content.")

    #return number_of_crash  # Optional, if you want to use the resampled data elsewhere

def plot_data_PERCLOS_LANE_DEVIATION(resampled_dfs, patient, session, Perclos_window_size, Lane_deviation_window_size, steering_wheel_std, plotting_activated):
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
            fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(16, 45), sharex=True)

            print(road_position_std.index)
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
                    ax2.set_ylabel('Heart Rate (BPM)', color='tab:blue')
                    ax2.tick_params(axis='y', labelcolor='tab:blue')
                    ax2.legend(loc='upper right')
                    ax2.set_title('Heart Rate and Filtered ECG Signal Over Time')

                    # Create a secondary y-axis for the normalized ECG signal
                    ax2b = ax2.twinx()
                    ax2b.plot(biopac_df.index / 60, normalized_ecg, color='tab:red', label='Filtered ECG Signal (Normalized)')
                    ax2b.set_ylabel('Filtered ECG Signal (Normalized)', color='tab:red')
                    ax2b.tick_params(axis='y', labelcolor='tab:red')
                    ax2b.legend(loc='upper left')

                    # Add markers at R peak indices
                    r_peak_times = biopac_df.index[r_peaks] / 60  # Convert to minutes
                    ax2b.plot(r_peak_times, normalized_ecg[r_peaks], 'o', color='tab:green', label='R Peaks')

                    # Adding legends and showing the plot
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


# Helper function for bandpass filtering
def bandpass_filter(data, lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data)
    return y

# Helper function for adaptive thresholding
def adaptive_thresholding(ecg_signal, initial_r_peaks, fs, window_size=0.75):
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
        threshold = np.mean(segment) + 3 * np.std(segment)
        
        # Find local maxima within the segment that exceed the threshold
        peaks, _ = find_peaks(segment, height=threshold)
        
        if len(peaks) > 0:
            # Select the peak closest to the center of the segment
            peak_distances = np.abs(peaks - (window_size_samples // 2))
            r_peak_candidate = peaks[np.argmin(peak_distances)] + start
            refined_r_peaks.append(r_peak_candidate)
            if math.isnan(r_peak_candidate):
                print("NAN FOUND")
    
    return np.array(refined_r_peaks)

def plot_ecg_signals(resampled_dfs, patient, session, plotting_activated):
    plotting = int(plotting_activated)
    full_session = f"{patient}_{session}"
    
    Map_quality_ECG_path = r'F:\Recordings\Qualité_signal_ECG.csv'
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
            filtered_ecg = bandpass_filter(ecg_signal, lowcut, highcut, fs)
            normalized_ecg = (filtered_ecg - np.min(filtered_ecg)) / (np.max(filtered_ecg) - np.min(filtered_ecg))
            normalized_ecg = normalized_ecg ** 2

            # Process the ECG signal to detect initial R peaks
            processed_ecg = nk.ecg_process(filtered_ecg, sampling_rate=fs)
            initial_r_peaks = processed_ecg[1]['ECG_R_Peaks']
            # Apply adaptive thresholding to refine R peaks
            refined_r_peaks = adaptive_thresholding(filtered_ecg, initial_r_peaks, fs)
            refined_r_peaks = verify_and_remove_duplicate_peaks(refined_r_peaks)
            # Calculate heart rate
            heart_rate = 60 / np.diff(refined_r_peaks) * fs
            hr_times = biopac_df.index[refined_r_peaks[1:]]
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
                        filtered_ecg_stm = bandpass_filter(ecg_signal_stm, lowcut, highcut, fs)
                        filtered_ecg_stm = filtered_ecg_stm ** 2
                        processed_ecg_stm = nk.ecg_process(filtered_ecg_stm, sampling_rate=fs)
                        initial_r_peaks_stm = processed_ecg_stm[1]['ECG_R_Peaks']
                        refined_r_peaks_stm = adaptive_thresholding(filtered_ecg_stm, initial_r_peaks_stm, fs)
                        refined_r_peaks_stm = verify_and_remove_duplicate_peaks(refined_r_peaks_stm)
                        heart_rate_stm = 60 / np.diff(refined_r_peaks_stm) * fs
                        hr_times_stm = STM32_ECG.index[refined_r_peaks_stm[1:]]
                        hr_series_stm = pd.Series(heart_rate_stm, index=hr_times_stm)
                        hr_series_list.append((hr_series_stm, column))
                        print(column)
                        selected_STM_ECG.append(column)
                        peak_indices_dict[column] = refined_r_peaks_stm  # Store peak indices
                        
            
            if plotting == 1:
                # Create subplots with shared x-axis
                fig, ax = plt.subplots(3, 1, figsize=(16, 30), sharex=True)
                # Plotting the heart rate for biopac
                
                ax[0].plot(biopac_df.index, normalized_ecg, color='tab:red', label='Biopac Filtered ECG Signal (Normalized)')
                ax[0].set_ylabel('Filtered ECG Signal (Normalized)', color='tab:red')
                ax[0].tick_params(axis='y', labelcolor='tab:red')
                ax[0].legend(loc='upper left')
                # Add markers at R peak indices
                r_peak_times = biopac_df.index[refined_r_peaks]
                ax[0].plot(r_peak_times, normalized_ecg[refined_r_peaks], 'o', color='tab:green', label='R Peaks')
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
                ax[1].legend(loc='upper right')
                ax[1].set_title('STM32 ECG Signals Over Time')
                for hr_series_stm, label in hr_series_list:
                    hr_series_corrected=advanced_bpm_correction(hr_series,hr_series_stm)

                 # Plotting the heart rate for biopac and STM32 signals
                ax[2].plot(hr_series.index, hr_series, label='Biopac Heart Rate (BPM)')
                ax[2].plot(hr_series_corrected.index, hr_series_corrected, label='Biopac Heart Rate corrected',linestyle='--')
                # Plotting the heart rate for STM32 signals
                for hr_series_stm, label in hr_series_list:
                    ax[2].plot(hr_series_stm.index, hr_series_stm, label=f'{label} Heart Rate (BPM)')

                ax[2].set_ylabel('Heart Rate (BPM)', color='tab:blue')
                ax[2].tick_params(axis='y', labelcolor='tab:blue')
                ax[2].legend(loc='upper right')
                ax[2].set_title('Heart Rate from All Signals Over Time')

                plt.tight_layout(pad=8.0)
                plt.subplots_adjust(hspace=0.4)
                plt.show()

    except KeyError as e:
        print(f"KeyError: {e}")
    except Exception as e:
        print(f"General error: {str(e)}")


# Function to verify and remove duplicate peaks
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

def advanced_bpm_correction(biopac_hr_series, stm32_hr_series, avg_window_size=60):
    # Create a common time base
    common_time_base = create_common_time_base(biopac_hr_series, stm32_hr_series)

    # Resample both Biopac and STM32 BPM series to the common time base
    resampled_biopac_hr_series = resample_bpm(biopac_hr_series, common_time_base)
    resampled_stm32_hr_series = resample_bpm(stm32_hr_series, common_time_base)

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
            (i > 0 and abs(stm32_bpm[i] - stm32_bpm[i - 1]) > 20)
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

# Example usage
# Assuming biopac_hr_series and stm32_hr_series are pandas Series of BPM values with datetime indices
# biopac_hr_series = pd.Series([...], index=pd.to_datetime([...]))
# stm32_hr_series = pd.Series([...], index=pd.to_datetime([...]))
# corrected_hr_series = advanced_bpm_correction(biopac_hr_series, stm32_hr_series)
