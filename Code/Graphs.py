import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

GROUPING_THRESHOLD = 0.3E-3

# Load the data
radiuses_df = pd.read_csv('Data/Radiuses.csv', header=None, names=['Particle', 'Radius'])
data_df = pd.read_csv('Data/Edited_Data.csv')
print("Data loaded")

# Round down the radii to the nearest 0.1E-3
radiuses_df['Radius_Group'] = (radiuses_df['Radius'] // GROUPING_THRESHOLD) * GROUPING_THRESHOLD
print("Radii rounded")

# Group particles by their radius group
grouped_radiuses = radiuses_df.groupby('Radius_Group')
print("Particles grouped by radius")

# Print the groups and their indices
print("Groups and their indices:")
for name, group in grouped_radiuses.groups.items():
    print(f"Group: {name}, Indices: {group}")

# Print the first few rows of each group
print("\nSnapshot of each group:")
for name, group in grouped_radiuses:
    print(f"\nGroup: {name}")
    print(group.head())

# Function to calculate average displacement squared over time
def calculate_avg_displacement_squared(group, data_df):
    particles = group['Particle'].values
    displacements_squared = []

    for particle in particles:
        x_col = f'Point-{particle} X'
        y_col = f'Point-{particle} Y'
        if x_col in data_df.columns and y_col in data_df.columns:
            x = data_df[x_col].values
            y = data_df[y_col].values
            displacement_squared = (x - x[0])**2 + (y - y[0])**2
            displacements_squared.append(displacement_squared)

    avg_displacement_squared = np.mean(displacements_squared, axis=0)
    return avg_displacement_squared

# Calculate and plot average displacement squared over time for each radius group
slopes = []
radius_groups = []

for radius_group, group in grouped_radiuses:
    avg_displacement_squared = calculate_avg_displacement_squared(group, data_df)
    time = data_df['time'].values

    # Filter out NaN values
    valid_indices = ~np.isnan(avg_displacement_squared)
    valid_time = time[valid_indices]
    valid_avg_displacement_squared = avg_displacement_squared[valid_indices]

    # Calculate the sample size as the number of particles in the group
    sample_size = len(group)

    if len(valid_time) < 2:
        print(f"Skipping group {radius_group:.1E} due to insufficient valid data points")
        continue

    plt.figure()
    plt.scatter(valid_time, valid_avg_displacement_squared, label=f'Radius Group: {radius_group:.1E}')
    
    # Calculate the slope and intercept of the fitted line
    slope, intercept = np.polyfit(valid_time, valid_avg_displacement_squared, 1)
    fitted_line = slope * valid_time + intercept
    
    # Plot the fitted line
    plt.plot(valid_time, fitted_line, label=f'Fitted Line: {radius_group:.1E}', linestyle='--')
    print(f"group {radius_group:.1E}: {slope:.2E}x + {intercept:.2E}")
    plt.xlabel('Time')
    plt.ylabel('Average Displacement Squared')
    plt.legend()
    upper_bound = radius_group + GROUPING_THRESHOLD
    plt.title(f'Avg Displacement Squared vs Time \n for Radius Group {radius_group:.1E}-{upper_bound:.1E} (Sample Size: {sample_size})')
    plt.savefig(f'Graphs/avg_displacement_squared_{radius_group:.1E}.png')
    # plt.show()  # Display the plot

    slopes.append(slope)
    radius_groups.append(radius_group)

# Plot the inverse of the slopes as a function of the particle radius
inverse_slopes = 1 / np.array(slopes)
plt.figure()
plt.scatter(radius_groups, inverse_slopes)
plt.xlabel('Particle Radius')
plt.ylabel('Inverse of Slope')
plt.savefig('Graphs/inverse_slopes.png')
# plt.show()  # Display the plot
