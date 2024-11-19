import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

GROUPING_THRESHOLD = 0.15E-6 #m
MAXIMUM_RADIUS = 2.8E-6 #m
PIXEL_SIZE = 0.102E-6 #m

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

# convert radiuses to meters from mm
radiuses_df['Radius'] = radiuses_df['Radius'] * 1E-3

# Convert pixel measurements to millimeters
for col in data_df.columns:
    if col.startswith('Point-') and (col.endswith('X') or col.endswith('Y')):
        data_df[col] = data_df[col] * PIXEL_SIZE

# Round down the radii to the nearest grouping threshold
radiuses_df['Radius_Group'] = (radiuses_df['Radius'] // GROUPING_THRESHOLD) * GROUPING_THRESHOLD
print("Radii rounded")

# Filter out groups with radius bigger than 
filtered_radiuses_df = radiuses_df[radiuses_df['Radius_Group'] < MAXIMUM_RADIUS]

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
sample_sizes = []
groups = []

plt.figure()

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
    
    if (sample_size >= 2):
        
        groups.append((radius_group, valid_time, valid_avg_displacement_squared, sample_size))

    # Plot individual graphs for each group
    plt.figure()
    plt.scatter(valid_time, valid_avg_displacement_squared, label=f'Group: {radius_group:.1E} (n={sample_size})')
    
    # Fit the line forcing the intercept to be zero
    A = valid_time[:, np.newaxis]  # Reshape time to be a 2D array
    slope, _, _, _ = np.linalg.lstsq(A, valid_avg_displacement_squared, rcond=None)
    slope = slope[0]
    fitted_line = slope * valid_time
    
    # Plot the fitted line
    plt.plot(valid_time, fitted_line, linestyle='--')
    print(f"group {radius_group:.1E}: {slope:.2E}x + 0")
    plt.xlabel('Time (s)')
    plt.ylabel('Average Displacement Squared (m^2)')
    plt.legend()
    plt.title(f'Avg Displacement Squared vs Time \n for Radius Group {radius_group:.1E} (Sample Size: {sample_size})')
    plt.savefig(f'{SAVE_FOLDER}/avg_displacement_squared_{radius_group:.1E}_{timestamp}.png')
    plt.close()

    slopes.append(slope)
    radius_groups.append(radius_group)
    sample_sizes.append(sample_size)

# Initialize a single figure for the unified graph
plt.figure()

# Assuming you have a list of groups with their respective data
for radius_group, valid_time, valid_avg_displacement_squared, sample_size in groups:
    # Plot the data points with smaller markers
    diff = radius_group + GROUPING_THRESHOLD
    plt.scatter(valid_time, valid_avg_displacement_squared, label=f'{radius_group:.2E}m-{diff:.2E}m (n={sample_size})', s=3)
    
    # Fit the line forcing the intercept to be zero
    A = valid_time[:, np.newaxis]  # Reshape time to be a 2D array
    slope, _, _, _ = np.linalg.lstsq(A, valid_avg_displacement_squared, rcond=None)
    slope = slope[0]
    fitted_line = slope * valid_time
    
    # Plot the fitted line with the same color as the data points
    plt.plot(valid_time, fitted_line, linestyle='--')
    #print(f"group {radius_group:.1E}: {slope:.2E}x + 0")

# Add labels, legend, and title to the unified graph
plt.xlabel('Time (s)')
plt.ylabel('Average Displacement Squared (m^2)')
plt.legend(fontsize='small')
plt.title('Avg Displacement Squared vs Time for All Radius Groups')

# Save the unified graph
plt.savefig(f'{SAVE_FOLDER}/avg_displacement_squared_all_groups_{timestamp}.png', dpi = 300)
plt.close()

# Filter out slopes with sample size less than two
filtered_slopes = []
filtered_radius_groups = []

for slope, radius_group in zip(slopes, radius_groups):
    group = grouped_radiuses.get_group(radius_group)
    sample_size = len(group)
    if sample_size >= 2:
        filtered_slopes.append(slope)
        filtered_radius_groups.append(radius_group)
    else:
        print(f"Skipping group {radius_group:.1E} due to insufficient sample size")
        

# Plot the inverse of the filtered slopes as a function of the particle radius
inverse_slopes = 1 / np.array(filtered_slopes)
plt.figure()

# Use a colormap to assign different colors to each point
cmap = plt.get_cmap('viridis')
colors = cmap(np.linspace(0, 1, len(filtered_radius_groups)))

# Plot each point with a different color
for i, radius_group in enumerate(filtered_radius_groups):
    plt.scatter(radius_group, inverse_slopes[i], color=colors[i], s=20, label=f'{radius_group:.2E}m')

# Fit a line to the data
coefficients = np.polyfit(filtered_radius_groups, inverse_slopes, 1)
fitted_line = np.polyval(coefficients, filtered_radius_groups)
plt.plot(filtered_radius_groups, fitted_line, linestyle='--', color='red', label=f'Fit: {coefficients[0]:.2E}x + {coefficients[1]:.2E}')

# Add legend
plt.legend()

plt.xlabel('Particle Radius (m)')
plt.ylabel('Inverse of Slope (s/m^2)')
plt.title('Inverse of Slope vs Particle Radius')
plt.savefig(f'{SAVE_FOLDER}/inverse_slopes_{timestamp}.png', dpi=300)
plt.close()
