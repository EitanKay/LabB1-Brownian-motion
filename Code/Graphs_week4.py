import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import warnings
import math

# Suppress FutureWarning
warnings.simplefilter(action='ignore', category=FutureWarning)


GROUPING_THRESHOLD = 0.15E-6 #m
MAXIMUM_RADIUS = 2.8 #m
MINIMUM_RADIUS = 1.2E-10 #m
PIXEL_SIZE = 0.0242 #m
RADIUS_ERROR = 0.1E-6 #m
MAX_WINDOW_SIZE = 25
MIN_SLOPE = 0

TEMP = 295 #K
PARTICLES_TO_PLOT = 1
FPS = 3
ERROR_TOLORENCE = 75

# LOCATION_DATA = ['Data/Edited_Data.csv',
#                  'Data/Week_2/outputs_10-percent_1/progress.csv',
#                  'Data/Week_2/outputs_20-percent_1/progress.csv',
#                  'Data/Week_2/outputs_30-percent_1/progress.csv',
#                  'Data/Week_2/outputs_40-percent_1/progress.csv',
#                  'Data/Week_2/outputs_50-percent_1/progress.csv']
# RADIUS_DATA = ['Data/Radiuses.csv',
#                'Data/Week_2/Radii.csv',
#                'Data/Week_2/Radii.csv',
#                'Data/Week_2/Radii.csv',
#                'Data/Week_2/Radii.csv',
#                'Data/Week_2/Radii.csv']\
LOCATION_DATA = ['Data/Week_4/data.csv']
RADIUS_DATA = ['Data/Week_4/radii.csv']

CATEGORY_NAMES = ['0%']

# Create a new folder in Graphs with timestamp
timestamp = time.strftime("%Y%m%d-%H%M%S")
SAVE_FOLDER = f'Graphs/{timestamp}'
os.makedirs(SAVE_FOLDER)

def load_data(location_data: str, radius_data:str):
    
# Load the data

    radiuses_df = pd.read_csv(radius_data, header=None, names=['Particle', 'Radius'])
    data_df = pd.read_csv(location_data)
    print(f"Data loaded, files: {location_data}, {radius_data}")

    # Convert radiuses to meters from mm
    # print(radiuses_df.head())
    radiuses_df['Radius'] = radiuses_df['Radius'] * 1E-2

    # Convert pixel measurements to millimeters
    for col in data_df.columns:
        if col.startswith('Point-') and (col.endswith('X') or col.endswith('Y')):
            data_df[col] = data_df[col] * PIXEL_SIZE
            
    return data_df, radiuses_df

# Function to calculate average displacement squared over time for a given window size
def calculate_avg_displacement_squared(particle, data_df, window_size):
    
    
    x_col = f'Point-{particle} X'
    y_col = f'Point-{particle} Y'
    if x_col in data_df.columns and y_col in data_df.columns:
        x = data_df[x_col].values
        y = data_df[y_col].values
        
        displacements_squared = []
        start = 0
        #print(f"WIndow size: {window_size}")
        while start + window_size < len(x):
            
            x_slice = x[start:(start + window_size)]
            y_slice = y[start:(start + window_size)]
  
            displacements_squared.append((x_slice[-1] - x_slice[0])**2 + (y_slice[-1] - y_slice[0])**2)
            start += window_size   
            
            #print(f"Displacement squared: {displacements_squared}")
            
        
        avg_displacement_squared = np.nanmean(displacements_squared)
        std_error = np.nanstd(displacements_squared) / np.sqrt(np.sum(~np.isnan(displacements_squared)))  # Standard error of the mean
        
        #print(f'Average displacement squared for particle {particle} with window size {window_size} is {avg_displacement_squared}')
        return avg_displacement_squared, std_error
    else:
        #print(f'Columns {x_col} or {y_col} not found in data for particle {particle}')
        return None

def plot_avg_displacement_squared(file_index: int):
    # Calculate and plot average displacement squared over time for each particle and window size
    LOCAL_SAVE_FOLDER = f"{SAVE_FOLDER}/{file_index}"
    os.makedirs(f"{LOCAL_SAVE_FOLDER}")
    data_df, radiuses_df = load_data(LOCATION_DATA[file_index], RADIUS_DATA[file_index])
    particles = radiuses_df['Particle'].values
    window_sizes = range(2, MAX_WINDOW_SIZE + 1)
    slopes_df = pd.DataFrame(columns=['Particle','Radius', 'Slope'])

    plt.figure()

    count = 0
    for particle in particles:
        radius = radiuses_df.loc[radiuses_df['Particle'] == particle, 'Radius'].values[0]
        
        #check if particle is in the data
        if f'Point-{particle} X' not in data_df.columns:
            #print(f'Particle {particle} not found in data')
            continue
        
        
        if radius > MAXIMUM_RADIUS or radius < MINIMUM_RADIUS:
            # print(f'Skipping particle {particle} with radius {radius} (greater than MAXIMUM_RADIUS)')
            continue
        # print(f'Calculating avg displacement squared for particle {particle} with radius {radius}')
        
        avg_displacements = []
        errors = []
        for window_size in window_sizes:
            #print(f'******************\nCalculating avg displacement squared for particle {particle} with window size {window_size}\n*********************')
            avg_displacement_squared, std_error = calculate_avg_displacement_squared(particle, data_df, window_size)
            #if avg_displacement_squared is not None:
            avg_displacements.append(avg_displacement_squared)
            errors.append(std_error)
        
        if avg_displacements:  # Ensure avg_displacements is not empty
        
            # fit a line to the data, force the intercept to be 0
            x = np.array(window_sizes[:len(avg_displacements)])
            y = np.array(avg_displacements)
            
            # Include the origin point (0, 0)
            x = np.insert(x, 0, 0)
            y = np.insert(y, 0, 0)
            
            
            # Reshape x for lstsq
            x = x[:, np.newaxis]
            
            # Use lstsq to fit the data with intercept forced to 0
            m, _, _, _ = np.linalg.lstsq(x, y, rcond=None)
            m = m[0]
            
            # calculate the error of the fit
            # Calculate the fitted values
            y_fit = m * x.flatten()
            
            # Calculate the residuals
            residuals = y - y_fit
            
            # Calculate the standard error of the regression
            residual_sum_of_squares = np.sum(residuals**2)
            degrees_of_freedom = len(y) - 1  # Number of observations minus number of parameters
            fit_error = np.sqrt(residual_sum_of_squares / degrees_of_freedom) / (np.sqrt(degrees_of_freedom+1))
            
            
            if count % PARTICLES_TO_PLOT == 0 and m and max(errors) < ERROR_TOLORENCE:
                
                scatter = plt.scatter(window_sizes[:len(avg_displacements)], avg_displacements, label=f'P: {particle}, R:{radius:.2e}', s=5)
                color = scatter.get_facecolor()[0]  # Get the color of the scatter plot
                plt.errorbar(window_sizes[:len(avg_displacements)], avg_displacements, yerr=errors, fmt='o', color=color, capsize=5)
                plt.plot(x, m*x, label=f'Fit: y = {m:.2e}x', linestyle='--', color=color)
            count += 1    
            
            # Add the slope to the dataframe
            if m and m > MIN_SLOPE and (max(errors) < ERROR_TOLORENCE):
                print(f'\033[92mParticle {particle} with slope {m:.2e} and fit error {fit_error:.2e} processed successfully\033[0m')
                new_row = pd.DataFrame({'Particle': [particle], 'Radius': [radius], 'Slope': [m*FPS], 'Fit Error': [fit_error]})
    
                # Check if new_row is empty or contains all-NA entries
                if not new_row.empty and not new_row.isna().all().all():
                    slopes_df = pd.concat([slopes_df, new_row], ignore_index=True)
                else:
                    print(f"Skipping empty or all-NA entry for particle {particle}")
            else:
                #print short red color message
                print(f'\033[91mSkipping particle {particle} with slope {m:.2e} and max error {max(errors):.2e}\033[0m')
            
        
        

    plt.xlabel(f'Frame (1/{FPS} s)')
    plt.ylabel('Average Displacement Squared (m^2)')
    plt.legend(fontsize='small')
    plt.title('Avg Displacement Squared vs time')
    plt.savefig(f'{LOCAL_SAVE_FOLDER}/avg_displacement_squared_all_particles_{timestamp}.png', dpi=300)

    # export the data to a csv
    slopes_df.to_csv(f'{LOCAL_SAVE_FOLDER}/slopes_{timestamp}-temp.csv', index=False)
    
    # Calculate 1/slope and add it to the dataframe
    slopes_df['Inverse Slope'] = 1 / slopes_df['Slope']
    slopes_df['Inverse Slope Error'] = slopes_df['Fit Error'] / (slopes_df['Slope'] ** 2)

    # delete the temporary csv
    os.remove(f'{LOCAL_SAVE_FOLDER}/slopes_{timestamp}-temp.csv')
    
    
    # Plot 1/slope as a function of the radius
    plt.figure()
    #plt.scatter(slopes_df['Radius'], slopes_df['Inverse Slope'], marker='o', linestyle='-')
    xerr = [RADIUS_ERROR] * len(slopes_df['Radius'])
    plt.errorbar(slopes_df['Radius'], slopes_df['Inverse Slope'], yerr=slopes_df ['Inverse Slope Error'], xerr=xerr, fmt='o', capsize=5)
    plt.errorbar(slopes_df['Radius'], slopes_df['Inverse Slope'], yerr=slopes_df ['Inverse Slope Error'], fmt='o', capsize=5)
    plt.xlabel('Radius (m)')
    plt.ylabel('1/Slope (s/m^2)')
    plt.title(f'1/Slope vs Radius for {file_index}0% glycerol')

    # Fit a line to the data with intercept forced to 0
    x = slopes_df['Radius'].values
    y = slopes_df['Inverse Slope'].values

    # Include the origin point (0, 0)
    x = np.insert(x, 0, 0)
    y = np.insert(y, 0, 0)
    # Reshape x for lstsq
    x = x[:, np.newaxis]

    # Use lstsq to fit the data with intercept forced to 0
    m, _, _, _ = np.linalg.lstsq(x, y, rcond=None)
    m = m[0]

    # Calculate the fitted values
    y_fit = m * x.flatten()
    
    # Calculate the residuals
    residuals = y - y_fit
    
    # Calculate the standard error of the regression
    residual_sum_of_squares = np.sum(residuals**2)
    degrees_of_freedom = len(y) - 1  # Number of observations minus number of parameters
    fit_error = (np.sqrt(residual_sum_of_squares / degrees_of_freedom) / (np.sqrt(np.sum((x - np.mean(x))**2)))) / np.sqrt(degrees_of_freedom+1)


    plt.plot(x, m*x, label=f'Fit: y = {m:.2e}x +- {fit_error:.2e}x', linestyle='--')
    plt.legend()
    plt.savefig(f'{LOCAL_SAVE_FOLDER}/inverse_slope_vs_radius_{timestamp}.png', dpi=300)
    #plt.show()
    plt.close()

    # Save the slopes dataframe to a csv
    slopes_df.to_csv(f'{LOCAL_SAVE_FOLDER}/slopes_{timestamp}.csv', index=False)
    
    # TODO - calculate the fit error properly
    return m, fit_error

def plot_slope_vs_viscosity(viscosity_slopes_df):
    
    # Plot slope vs viscosity
    plt.figure()
    plt.scatter(viscosity_slopes_df['Viscosity'],viscosity_slopes_df['Slope'])
    plt.errorbar(viscosity_slopes_df['Viscosity'], viscosity_slopes_df['Slope'], yerr=viscosity_slopes_df['Fit Error'], fmt='o', capsize=5)
    # Fit a line to the data with intercept forced to 0
    x = viscosity_slopes_df['Viscosity'].values
    y = viscosity_slopes_df['Slope'].values

    # Include the origin point (0, 0)
    x = np.insert(x, 0, 0)
    y = np.insert(y, 0, 0)

    # Reshape x for lstsq
    x = x[:, np.newaxis]
    
    

    # Use lstsq to fit the data with intercept forced to 0
    m, _, _, _ = np.linalg.lstsq(x, y, rcond=None)
    m = m[0]

    # Calculate the fitted values
    y_fit = m * x.flatten()
    
    # Calculate the residuals
    residuals = y - y_fit
    
    # Calculate the standard error of the regression
    residual_sum_of_squares = np.sum(residuals**2)
    
    x_variance = np.var(x, ddof=1)

    degrees_of_freedom = len(y) - 1  # Number of observations minus number of parameters
    
    s_yx = np.sqrt(residual_sum_of_squares / degrees_of_freedom)
    standard_error_m = (s_yx / np.sqrt(np.sum((x - np.mean(x))**2))) / np.sqrt(degrees_of_freedom+1)

    plt.plot(x, y_fit, label=f'Fit: y = {m:.2e}x +- {standard_error_m:.2e}x', linestyle='--')
    plt.legend()
    
    plt.xlabel('Viscosity (Pa s)')
    plt.ylabel('S_2 (Pa s / J)')
    plt.title('S_2 vs Viscosity')
    plt.savefig(f'{SAVE_FOLDER}/slope_vs_viscosity_{timestamp}.png', dpi=300)
    plt.close()  
    
    #calculate the boltzmann constant
    boltzmann_constant = (3*(math.pi)) / (2 * TEMP *m)
    return boltzmann_constant
    
    

def main():
    
    # Create a dataframe to store the slopes and viscosities
    viscosities = pd.read_csv('Data/Viscosities.csv')
    
    # add a slope column to the dataframe
    viscosities['Slope'] = np.nan
    viscosities['Fit Error'] = np.nan
    
    
    for file_index in range(len(LOCATION_DATA)):
        slope, fit_error = plot_avg_displacement_squared(file_index)
        viscosities.at[file_index, 'Slope'] = slope
        viscosities.at[file_index, 'Fit Error'] = fit_error
        #print(viscosities.head())
       
        print(f"Plotted data for {LOCATION_DATA[file_index]}")
        
    # Save the viscosities dataframe to a csv
    viscosities.to_csv(f'{SAVE_FOLDER}/viscosities_{timestamp}.csv', index=False)
    
    # Plot slope vs viscosity    
    calculated_BC = plot_slope_vs_viscosity(viscosities)
    print(f"Calculated Boltzmann Constant: {calculated_BC}")

    print('')

        
        
if __name__ == '__main__':
    main()