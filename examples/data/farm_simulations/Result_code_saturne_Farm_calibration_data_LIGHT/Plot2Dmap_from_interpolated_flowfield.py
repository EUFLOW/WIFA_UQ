import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset

# Path to the NetCDF file
FILE_PATH = "farm_result/flow_field_cs_run_staggered_DX9D_DY5D_Turbine_NumberNumber100.nc"

def plot_wind_speed(case_number):
    """
    Plot the wind speed map for a given case number.

    Parameters:
    - case_number (int): The case number to plot.
    """
    # Open the NetCDF file
    dataset = Dataset(FILE_PATH, "r")

    # Read dimensions and variables
    x = dataset.variables["x"][:]
    y = dataset.variables["y"][:]
    wind_speed = dataset.variables["wind_speed"][:]

    # Create a meshgrid for plotting
    X, Y = np.meshgrid(x, y)

    # Extract wind speed data for the selected case at altitude 0
    wind_speed_case = wind_speed[case_number, 0, :, :].T

    # Plot the wind speed map
    plt.figure(figsize=(10, 6))
    contour = plt.contourf(X, Y, wind_speed_case, levels=1000, cmap='turbo')
    plt.colorbar(contour, label="Wind Speed (m/s)")
    plt.title(f"Wind Speed for Case {case_number}")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.show()

    # Close the NetCDF file
    dataset.close()


if __name__ == "__main__":
        # Example usage: plot wind speed for case number 5
    case_number = 5  # You can change this value to plot a different case
    plot_wind_speed(case_number)
