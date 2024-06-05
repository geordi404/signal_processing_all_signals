import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description='Process the path, patient number, and session for CSV data analysis.')
    parser.add_argument('-d', '--directory', required=True, help='Base directory path where recordings are stored')
    parser.add_argument('-p', '--patient', required=True, help='Patient number (e.g., P001, P002)')
    parser.add_argument('-s', '--session', required=True, help='Session analyzed (e.g., S001)')
    parser.add_argument('-plt', '--plotting_activated', required=True, help='1 = plotting, 0 = not_plotting')
    parser.add_argument('-c', '--calculate_perclos', required=True, help='1 = calculate, 0 = dont calculate')


    args = parser.parse_args()

    return args