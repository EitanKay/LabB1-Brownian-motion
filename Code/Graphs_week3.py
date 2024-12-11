import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import warnings
import math
from scipy.optimize import curve_fit # type: ignore

# Suppress FutureWarning
warnings.simplefilter(action='ignore', category=FutureWarning)


GROUPING_THRESHOLD = 0.15E-6 #m
MAXIMUM_RADIUS = 1 #m
MINIMUM_RADIUS = 1.2E-6 #m
PIXEL_SIZE = 0.0242 #m
RADIUS_ERROR = 0.1E-6 #m
MAX_WINDOW_SIZE = 20
MINIMUM_DATA_POINTS = 200

TEMP = 295 #K
PARTICLES_TO_PLOT = 3
FPS = 29.86
# ERROR_TOLORENCE = 75E-14
ERROR_TOLORENCE  = 1

LOCATION_DATA = ['Data/Week_3/Week3 data.csv']
RADIUS_DATA = ['Data/Week_3/radii_Week3.csv']

CATEGORY_NAMES = ['0%']

# Create a new folder in Graphs with timestamp
timestamp = time.strftime("%Y%m%d-%H%M%S")
SAVE_FOLDER = f'Graphs/{timestamp}'
os.makedirs(SAVE_FOLDER)

def load_data(location_data: str, radius_data:str):
    
# Load the data

    radiuses_df = pd.read_csv(radius_data, header=None, names=['Particle', 'Radius', 'Description'])
    data_df = pd.read_csv(location_data)
    print(f"Data loaded, files: {location_data}, {radius_data}")

    # Convert radiuses to meters from mm
    print(radiuses_df.head())
    radiuses_df['Radius'] = radiuses_df['Radius'] * 1E-2

    # Convert pixel measurements to millimeters
    for col in data_df.columns:
        if col.startswith('Point-') and (col.endswith('X') or col.endswith('Y')):
            data_df[col] = data_df[col] * PIXEL_SIZE
    
    print(data_df.head())     
    return data_df, radiuses_df

# Function to calculate average displacement squared over time for a given window size
def calculate_avg_displacement_squared(particle, data_df, window_size):
    x_col = f'Point-{particle} X'
    y_col = f'Point-{particle} Y'
    if x_col not in data_df.columns or y_col not in data_df.columns:
        return None
    
    
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
    

def linear_func_no_intercept(x, m):
    """This function is used to fit a line to the data with the intercept forced to 0
    """
    return m * x

def linear_func(x, m, c):
    """This function is used to fit a line to the data
    """
    return m * x + c

def gaussian_func(x, m, s):
    """
    Calculate the Gaussian function values for the input array x.

    Parameters:
        m (float): Mean of the Gaussian distribution.
        s (float): Standard deviation of the Gaussian distribution.
        x (numpy.ndarray): Input array of x values.

    Returns:
        numpy.ndarray: Values of the Gaussian function for the input x.
    """
    return (1 / (np.sqrt(2 * np.pi) * s)) * np.exp(-0.5 * ((x - m) / s) ** 2)

            
def plot_avg_displacement_squared(file_index: int):
    # Calculate and plot average displacement squared over time for each particle and window size
    LOCAL_SAVE_FOLDER = SAVE_FOLDER
    # os.makedirs(f"{SAVE_FOLDER}")
    data_df, radiuses_df = load_data(LOCATION_DATA[file_index], RADIUS_DATA[file_index])
    print(radiuses_df)
    particles = radiuses_df['Particle'].values
    window_sizes = range(2, MAX_WINDOW_SIZE + 1)
    slopes_df = pd.DataFrame(columns=['Particle','Radius', 'Slope'])

    count = 0
    for particle in particles:
        radius = radiuses_df.loc[radiuses_df['Particle'] == particle, 'Radius'].values[0]

        desc = radiuses_df.loc[radiuses_df['Particle'] == particle, 'Description'].values[0]
        
        # Check if particle is in the data
        if f'Point-{particle} X' not in data_df.columns:
            print(f"\033[91mParticle {particle} is not in the data\033[0m")
            continue
        
        if radius > MAXIMUM_RADIUS or radius < MINIMUM_RADIUS:
            print(f"\033[91mParticle {particle} is not in the radius frame\033[0m")
            continue
        
        coumn_name = f'Point-{particle} X'
        non_nan_count = data_df[coumn_name].dropna().count()
        #print(f"has {non_nan_count} non-nan values")
        if non_nan_count < MINIMUM_DATA_POINTS:
            print(f'Skipping particle {particle} with insufficient data points, had {data_df[f"Point-{particle} X"].dropna().count()}')
            continue
        # gaussian(data_df[f"Point-{particle} X"], f"{desc}, x axis", LOCAL_SAVE_FOLDER)
        # gaussian(data_df[f"Point-{particle} Y"], f"{desc}, y axis", LOCAL_SAVE_FOLDER)
        avg_displacements = []
        errors = np.array([])
        for window_size in window_sizes:
            
            avg_displacement_squared, std_error = calculate_avg_displacement_squared(particle, data_df, window_size)
        #print(f'******************\nCalculating avg displacement squared for particle {particle} with window size {window_size}\n*********************')
            
            #if avg_displacement_squared is not None:
            avg_displacements.append(avg_displacement_squared)
            #errors.append(std_error)
            errors = np.append(errors, std_error)
        
        if not avg_displacements:
            continue  
         
        # fit a line to the data, force the intercept to be 0
        x = np.array(window_sizes[:len(avg_displacements)]) -1
        #print(x)
        y = np.array(avg_displacements)

        x = np.insert(x,0,0)
        y = np.insert(y,0,0)
        errors = np.insert(errors,0,0)

        # Remove entries where y is NaN
        mask = ~np.isnan(y)
        x = x[mask]
        y = y[mask]
        errors = errors[mask]

        
        # assert to check for infinite values
        
        initial_guess = [1E-4,0]
        a_fit, cov = curve_fit(linear_func, x, y, nan_policy='omit', p0=initial_guess, maxfev=100000)
        
        
        m = a_fit[0]
        intercept = a_fit[1]
        fit_error = np.sqrt(cov[0, 0])
        
        
        # Add the slope to the dataframe
        # if m and m > 1E-14 and (max(errors) < ERROR_TOLORENCE) and not np.isinf(fit_error) and fit_error < 1E-14 and abs(intercept) < MAX_INTERCEPT:
        # if count < PARTICLES_TO_PLOT:
        plt.figure()
        plt.scatter(x,y, label=f'P: {particle}, R:{radius:.2e}', s=5)
        # scatter.get_facecolor()[0]  # Get the color of the scatter plot
        plt.errorbar(x,y, yerr=errors, fmt='o', capsize=5)
        plt.plot(x, m*x + intercept, linestyle='--')
        plt.xlabel(f'Frame (1/{FPS} s)')
        plt.ylabel('Average Displacement Squared (m^2)')
        plt.legend(fontsize='small')
        plt.title(f'Avg Displacement Squared vs time\nfor {desc}')
        plt.savefig(f'{LOCAL_SAVE_FOLDER}/{desc}.png', dpi=300)

        count += 1 
        print(f'\033[92mParticle {particle} with slope {m:.2e} and max error {max(errors):.2e} processed successfully\033[0m')
        new_row = pd.DataFrame({'Particle': [particle], 'Radius': [radius], 'Slope': [m*FPS], 'Fit Error': [fit_error]})
        
        count += 1    
        
        # Add the slope to the dataframe
        # if m and m > 1.7E-14 and (max(errors) < ERROR_TOLORENCE):
        print(f'\033[92mParticle {particle} with slope {m:.2e} and fit error {fit_error:.2e} processed successfully\033[0m')
        new_row = pd.DataFrame({'Particle': [particle], 'Radius': [radius], 'Slope': [m*FPS], 'Fit Error': [fit_error]})

        # Check if new_row is empty or contains all-NA entries
        if not new_row.empty and not new_row.isna().all().all():
            slopes_df = pd.concat([slopes_df, new_row], ignore_index=True)
        else:
            print(f"Skipping empty or all-NA entry for particle {particle}")
        # else:
        #     print(f'\033[91mSkipping particle {particle} with slope {m:.2e} and max error {max(errors):.2e}\033[0m')
        

        # Check if new_row is empty or contains all-NA entries
        if not new_row.empty and not new_row.isna().all().all():
            slopes_df = pd.concat([slopes_df, new_row], ignore_index=True)
        else:
            print(f"Skipping empty or all-NA entry for particle {particle}")
        # else:
        #     #print short red color message
            print(f'\033[91mSkipping particle {particle} with slope {m:.2e} and max error {max(errors):.2e} and intercept {intercept} \033[0m')
        

def gaussian(data, desc, LOCAL_SAVE_FOLDER):
    print(data)
    data = [d for d in data if not np.isnan(d)]
    i = 1
    distances = {}
    while i < len(data):
        dist = round(data[i-1] - data[i], 10)
        # print(dist)
        if not dist in distances:
            distances[dist] = 1
        else:
            distances[dist] += 1
        i+= 2
        
    
    x = np.array(list(distances.keys()))
    y = np.array(list(distances.values()))
    print("x: ",x,"y: ", y)
    
    initial_guess = [np.mean(x), np.std(x)]
    
    
    a_fit, cov = curve_fit(gaussian_func, x, y,p0=initial_guess, nan_policy='omit')
    
    print(a_fit)
    m = a_fit[0]
    s = a_fit[1]
    print("GAUSSIAN: ", gaussian_func(x, m, s))
    print("m: ", m,"s: ",s)
    print(f"mean: {np.mean(x)}, std: {np.std(x)}")
    fit_error = np.sqrt(cov[0, 0])
    
    plt.figure()
    plt.bar(x, y, width=0.0001)

    g = gaussian_func(x, m, s)
    plt.scatter(x, g)
    left = np.abs(max(-0.01,min(x)))
    right = min(0.01, max(x))
    lim = max(left, right)
    plt.xlim(left=-lim , right=lim)
    
    j = 0
    while j < len(x):
        print(f"gaussian for {x[j]}: {g[j]}")
        j+= 1

    # plt.title(desc)
    plt.xlabel("distance (cm)")
    plt.ylabel("no. of occurrences")
    plt.savefig(f'{LOCAL_SAVE_FOLDER}/{desc}.png', dpi=300)
    plt.close()

def main():
    
    # Create a dataframe to store the slopes and viscosities
    viscosities = pd.read_csv('Data/Viscosities.csv')
    
    # add a slope column to the dataframe
    viscosities['Slope'] = np.nan
    viscosities['Fit Error'] = np.nan
    
    
    for file_index in range(len(LOCATION_DATA)):
        plot_avg_displacement_squared(file_index)
        
    # Save the viscosities dataframe to a csv
    viscosities.to_csv(f'{SAVE_FOLDER}/viscosities_{timestamp}.csv', index=False)
    
    # Plot slope vs viscosity    
    # calculated_BC = plot_slope_vs_viscosity(viscosities)
    # print(f"Calculated Boltzmann Constant: {calculated_BC}")

    print('')

        
        
if __name__ == '__main__':
    main()