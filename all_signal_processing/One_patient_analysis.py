import pandas as pd
import os
from Arg_parse import parse_arguments
from Data_processing import Data_processing
from Calculate_new_perclos import Calculate_new_perclos

def main():
    # Setting up ArgumentParser to read command line arguments
    args, args_dict =parse_arguments()
    full_session = f"{args.patient}_{args.session}"
    directory_path = os.path.join(args.directory, args.patient, full_session, f"{full_session}_aligned")
    csv_file = f'{full_session}_combined_data.csv'
    file_path = os.path.join(directory_path, csv_file)
    #windowsizes=[15,30,45,60,75,90,105,120,135,150,165,180,195,210,225,240,255,270,285,300]
    windowsizes=[60]
    Patients_numbers_crash=[]
    # Call the function with command-line arguments
    if 0==int(args.calculate_perclos):
        for Window_size in windowsizes:
            
            Patients_numbers_crash.append(Data_processing(args.directory, args_dict, args.patient, args.session,0.25,Window_size,Window_size,args.plotting_activated))
            

    elif 1==int(args.calculate_perclos):


        df_new_perclos=Calculate_new_perclos(args.directory, args.patient, args.session,0.25,60,0)

    print(f"#########{Patients_numbers_crash}#########")
    return Patients_numbers_crash
    

        

if __name__ == '__main__':
    main()
