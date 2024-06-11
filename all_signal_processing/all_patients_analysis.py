import os
import subprocess
import re
from multiprocessing import Process, Manager

def check_filename(filename):
    pattern = r"P\d+_S\d+"
    return bool(re.search(pattern, filename))

def split_sp_code(sp_code):
    parts = sp_code.split('_')
    return parts[0], parts[1]

def analyze_patient_range(base_path, patient_range, return_dict):
    for i in patient_range:
        temporary_data=[]
        patient_id = f'P{str(i).zfill(3)}'
        patient_path = os.path.join(base_path, patient_id)

        if not os.path.exists(patient_path):
            print(f"Directory {patient_path} not found, skipping...")
            continue
        print(f"Entering Directory: {patient_path}")

        for patient_level_dir in os.listdir(patient_path):
            session_path = os.path.join(patient_path, patient_level_dir)
            if not os.path.isdir(session_path):
                continue
            full_session_path = os.path.join(session_path, f'{patient_level_dir}_aligned')

            if check_filename(full_session_path):
                print(f"Entering directory: {full_session_path}")
                Pxxx, Sxxx = split_sp_code(patient_level_dir)
                print(f"Analyzing {Pxxx}_{Sxxx}...")
                command = ['python', 'One_patient_analysis.py', '-d', base_path, '-p', Pxxx, '-s', Sxxx, '-c', str(0),'-plt',str(1), '-ecg', str(0), '-rsp', str(1), '-ptype', 'html']
                try:
                    result = subprocess.run(command, check=True, capture_output=True, text=True)
                    print(f"successfully ran {Pxxx}_{Sxxx}!")
                    temporary_data.append(result.stdout)
                    
                except subprocess.CalledProcessError as e:
                    print(f"Failed to execute command for {patient_level_dir}: {e}")
                    return_dict[i] = f"Failed: {e}"

        return_dict[i] = temporary_data

def run_program(base_path):
    manager = Manager()

    return_dict = manager.dict()

    ranges = [
        range(1, 6),
        range(6, 11),
        range(11, 16),
        range(16, 21),
        range(21, 26)
    ]

    processes = []
    for patient_range in ranges:
        p = Process(target=analyze_patient_range, args=(base_path, patient_range, return_dict))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    sorted_results = {k: return_dict[k] for k in sorted(return_dict)}
    for patient, results in sorted_results.items():
        for result in results:
            print(f"Patient {patient}: {result}########################################################")

if __name__ == '__main__':
    base_directory = r'E:\IRSST_recordings\Recordings'
    run_program(base_directory)
