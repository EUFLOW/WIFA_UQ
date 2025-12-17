import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset

# Nom du fichier NetCDF dans le répertoire courant
file_name = "single_time_flow_field_interpolated.nc"

# Specific case to plot (e.g., case 1)
specific_case_index = 0

# Open the NetCDF file
dataset = Dataset(file_name, "r")

# Read dimensions and variables
x = dataset.variables["x"][:]
y = dataset.variables["y"][:]
time = dataset.variables["time"][:]
wind_speed = dataset.variables["wind_speed"][:]
X, Y = np.meshgrid(x, y)

# Select the specific case
wind_speed_case = wind_speed[specific_case_index, 0, :, :]  # 0 for altitude

# Define the x positions for which to plot the profiles
x_positions = [1200, 2400, 3600]  # Corresponding to x=5D, x=10D, x=15D

# Plot profiles parallel to the x-axis at different y positions
for x_pos in x_positions:
    x_index = np.abs(x - x_pos).argmin()  # Find the index closest to the x position
    wind_speed_profile = wind_speed_case[x_index, :]

    plt.figure(figsize=(10, 6))
    plt.plot(y, wind_speed_profile, "+", label=f"x = {x_pos} m")

    # Set plot title and labels
    plt.title(f"Wind Speed Profile at x = {x_pos} m for Case {specific_case_index + 1}")
    plt.xlabel("Y (m)")
    plt.ylabel("Wind Speed (m/s)")
    plt.legend()
    plt.show()

# Plot profile along y=0 from x = -1000 to x = 5000
y_index = np.abs(y - 0).argmin()  # Find the index closest to y = 0
wind_speed_profile_y0 = wind_speed_case[:, y_index]

plt.figure(figsize=(10, 6))
plt.plot(x, wind_speed_profile_y0, "+", label="y = 0 m")

# Set plot title and labels
plt.title(f"Wind Speed Profile at y = 0 m for Case {specific_case_index + 1}")
plt.xlabel("X (m)")
plt.ylabel("Wind Speed (m/s)")
plt.legend()
plt.show()

# Close the NetCDF file
dataset.close()
