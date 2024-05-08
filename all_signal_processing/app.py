import pandas as pd
import os
from Arg_parse import parse_arguments
from Data_processing import Data_processing


def main():
    # Setting up ArgumentParser to read command line arguments
    args =parse_arguments()
    # Call the function with command-line arguments
    Data_processing(args.directory, args.patient, args.session)

if __name__ == '__main__':
    main()
