import netCDF4 as nc
import matplotlib.pyplot as plt
import pandas as pd

# Open the NetCDF file
file_path = "single_time_flow_field.nc"
dataset = nc.Dataset(file_path)

# Read the variables
x = dataset.variables["x"][:]
y = dataset.variables["y"][:]
time = dataset.variables["time"][:]
wind_speed = dataset.variables["wind_speed"][:]

# Close the NetCDF file
dataset.close()


# Function to create a mask within a given interval
def create_mask(array, value, interval):
    return (array >= value - interval) & (array <= value + interval)


# Define the points of interest with intervals
interval = 30  # Interval of 30 meters
case_number = 0  # Replace this value with the desired case number

points_of_interest = {
    "y=0": (create_mask(y, 0, interval), (x >= -1000) & (x <= 5000)),
    "x=5D": (create_mask(x, 1200, interval), (y >= -2000) & (y <= 2000)),
    "x=10D": (create_mask(x, 2400, interval), (y >= -2000) & (y <= 2000)),
    "x=15D": (create_mask(x, 3600, interval), (y >= -2000) & (y <= 2000)),
}

# Plot the wind speed profiles
for label, (mask_y, mask_x) in points_of_interest.items():
    mask = mask_y & mask_x
    selected_x = x[mask]
    selected_y = y[mask]
    selected_speed = wind_speed[mask, 0, case_number]

    df = pd.DataFrame(
        {"coord": selected_x if "y=0" in label else selected_y, "speed": selected_speed}
    )

    plt.figure()
    plt.plot(df["coord"], df["speed"], "+", label=label)
    plt.xlabel("x [m]" if "y=0" in label else "y [m]")
    plt.ylabel("Wind Speed")
    plt.title(f"Wind Speed Profile for {label}")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.show()
