import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Read the CSV file into a DataFrame
file_path = 'F:/Recordings/data_analysis/crash.csv'
df = pd.read_csv(file_path)

# Separate the DataFrame into two based on the 'PM_Soir' column
df_pm = df[df['PM_Soir'] == 'PM']
df_soir = df[df['PM_Soir'] == 'Soir']

def add_std_brackets(ax, data, positions):
    for pos, col in zip(positions, data.columns):
        std_dev = data[col].std()
        mean = data[col].mean()
        

def plot_combined_boxplot_with_std(data_pm, data_soir, columns, title_pm, title_soir):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Plot for PM category
    sns.boxplot(data=data_pm[columns], ax=axes[0])
    add_std_brackets(axes[0], data_pm[columns], range(len(columns)))
    axes[0].set_title(title_pm)
    axes[0].set_ylabel("Nombre d'accidents")

    # Plot for Soir category
    sns.boxplot(data=data_soir[columns], ax=axes[1])
    add_std_brackets(axes[1], data_soir[columns], range(len(columns)))
    axes[1].set_title(title_soir)
    axes[1].set_ylabel("Nombre d'accidents")

    plt.tight_layout()
    plt.show()

# Columns to plot
half_columns = ['Session 1', 'Session 2']

# Plot combined figure for both categories
plot_combined_boxplot_with_std(df_pm, df_soir, half_columns, 
                               "Expérience type 1: Distribution des accidents en fonction du matin et de l'après-midi",
                               "Expérience type 2: Distribution des accidents en fonction du matin et du soir")
