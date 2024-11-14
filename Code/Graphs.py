import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

GROUPING_THRESHOLD = 0.2E-3
#create a new folder in Graphs with timestamp
import os
import time
timestamp = time.strftime("%Y%m%d-%H%M%S")
SAVE_FOLDER = f'Graphs/{timestamp}'
os.makedirs(SAVE_FOLDER)


# Load the data
radiuses_df = pd.read_csv('Data/Radiuses.csv', header=None, names=['Particle', 'Radius'])
data_df = pd.read_csv('Data/Edited_Data.csv')
print("Data loaded")

# Round down the radii to the nearest 0.1E-3
radiuses_df['Radius_Group'] = (radiuses_df['Radius'] // GROUPING_THRESHOLD) * GROUPING_THRESHOLD
print("Radii rounded")

# Filter out groups with radius bigger than 0.3E-3
filtered_radiuses_df = radiuses_df[radiuses_df['Radius_Group'] < 2.8E-3]

# Group particles by their radius group
grouped_radiuses = filtered_radiuses_df.groupby('Radius_Group')
print("Particles grouped by radius")

# Print the groups and their indices
print("Groups and their indices:")
for name, group in grouped_radiuses:
    print(f"Group: {name}, Indices: {group.index.tolist()}")

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
    
    # Fit the line forcing the intercept to be zero
    A = valid_time[:, np.newaxis]  # Reshape time to be a 2D array
    slope, _, _, _ = np.linalg.lstsq(A, valid_avg_displacement_squared, rcond=None)
    slope = slope[0]
    fitted_line = slope * valid_time
    
    # Plot the fitted line
    plt.plot(valid_time, fitted_line, label=f'Fitted Line: {radius_group:.1E}', linestyle='--')
    print(f"group {radius_group:.1E}: {slope:.2E}x + 0")
    plt.xlabel('Time')
    plt.ylabel('Average Displacement Squared')
    plt.legend()
    upper_bound = radius_group + GROUPING_THRESHOLD
    plt.title(f'Avg Displacement Squared vs Time \n for Radius Group {radius_group:.1E}-{upper_bound:.1E} (Sample Size: {sample_size})')
    plt.savefig(f'{SAVE_FOLDER}/avg_displacement_squared_{radius_group:.1E}_{timestamp}.png')
    plt.close()

    slopes.append(slope)
    radius_groups.append(radius_group)

# Plot the inverse of the slopes as a function of the particle radius
inverse_slopes = 1 / np.array(slopes)
plt.figure()
plt.scatter(radius_groups, inverse_slopes)

# Add labels to each point
for i, radius_group in enumerate(radius_groups):
    plt.text(radius_group, inverse_slopes[i], f'{radius_group:.1E}', fontsize=9, ha='right')

plt.xlabel('Particle Radius')
plt.ylabel('Inverse of Slope')
plt.savefig(f'{SAVE_FOLDER}/inverse_slopes_{timestamp}.png')
# plt.show()  # Display the plot
