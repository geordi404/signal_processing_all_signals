import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description='Process the path, patient number, and session for CSV data analysis.')
    parser.add_argument('-d', '--directory', required=True, help='Base directory path where recordings are stored')
    parser.add_argument('-p', '--patient', required=True, help='Patient number (e.g., P001, P002)')
    parser.add_argument('-s', '--session', required=True, help='Session analyzed (e.g., S001)')
    parser.add_argument('-plt', '--plotting_activated', required=True, help='1 = plotting, 0 = not_plotting')
    parser.add_argument('-ptype', '--plot_type', required=True, help='html to save plot in html format, matplotlib to show plot in matplotlib')
    parser.add_argument('-rsp', '--respiration', required=True, help='1 = respiration, 0 = no_respiration')
    parser.add_argument('-ecg', '--ecg', required=True, help='1 = heart_rate, 0 = no_heart_rate')
    parser.add_argument('-c', '--calculate_perclos', required=True, help='1 = calculate, 0 = dont calculate')

    args = parser.parse_args()

    arg_dict = {
        'directory': args.directory,
        'patient': args.patient,
        'session': args.session,
        'plotting_activated': args.plotting_activated,
        'calculate_perclos': args.calculate_perclos,
        'plot_type': args.plot_type,
        'respiration': args.respiration,
        'ecg': args.ecg
    }
    return args, arg_dict