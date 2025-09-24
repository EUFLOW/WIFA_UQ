import os
import pandas as pd
import matplotlib.pyplot as plt

def plot_values(directory, value_to_plot, file_numbers):
    plt.figure(figsize=(12, 8))

    # Iterate over the specified file numbers
    for file_number in file_numbers:
        file_name = f'meteo_file_{file_number:06d}.csv'
        file_path = os.path.join(directory, file_name)

        # Read the CSV file
        df = pd.read_csv(file_path)

        # Plot the selected value against altitude
        plt.plot(df[value_to_plot].values,df['Altitude'].values, label=f'File {file_number}')

    # Configure the plot
    plt.ylabel('Altitude (m)')
    plt.ylim(0,2000)
    plt.xlabel(value_to_plot)
    plt.title(f'Comparison of {value_to_plot} vs Altitude')
    plt.legend()
    plt.grid(True)
    plt.show()

# Example usage
directory = 'MeteoFiles'  # Replace with the actual directory containing the CSV files
value_to_plot = 'u'  # Replace with the value you want to plot (Temperature, U, V, K, Eps)
file_numbers = [1,3,5,7,9]  # Replace with the file numbers you want to compare

plot_values(directory, value_to_plot, file_numbers)

