import os
import csv
from scipy.io import loadmat

import pandas as pd

# Specify the paths
lat_lon_combined_path = '/share/wildfire-2/sohel/Project/output_csv/lat_lon_combined.csv'
input_folder_path = '/share/wildfire-2/sohel/Project/03_Sampling/result/Wholeimage_RF/Case_num_119/trial_Final'
output_folder_path = '/share/wildfire-2/sohel/Project/csv_latlon/119/Scheme2/new'

# Extract fire / nonfire samples
path_tj = '/share/wildfire-2/tjsung/Research/GK2A_ActiveFire_South_Korea/'
path_sh = '/share/wildfire-2/sohel/Project/'

# Add the preprocessing path to the Python path
preprocessing_path = os.path.join(path_tj, '03_Preprocessing/')
os.sys.path.append(preprocessing_path)

# Read grids
grid_path = os.path.join(path_sh, '01_Data/Grid')

# Load GK2A lonlat data from MATLAB file
lonlat_data = loadmat(os.path.join(grid_path, 'GK2A_KO_2km_lonlat.mat'))
lat_2km = lonlat_data['lat_2km']
lon_2km = lonlat_data['lon_2km']

# Reshape the variable to (81000,)
reshaped_lat = lat_2km.flatten()
reshaped_lon = lon_2km.flatten()

input_folder_path = '/share/wildfire-2/sohel/Project/03_Sampling/result/Wholeimage_RF/Case_num_119/trial_Final/scheme1_20220304_0218_input_results.csv'
# Read the input CSV file
input_data = pd.read_csv(input_file_path)
# Combine lat and lon into a single list of tuples
combined_data = list(zip(reshaped_lat, reshaped_lon,input_data))
print(combined_data)




# Loop through each file in the input folder
for filename in os.listdir(input_folder_path):
    if filename.endswith('.csv') and 'scheme2' in filename:
        # Construct the full path of the input CSV file
        input_file_path = os.path.join(input_folder_path, filename)

        # Read the input CSV file
        input_data = pd.read_csv(input_file_path)
        # Combine lat and lon into a single list of tuples
        combined_data = list(zip(reshaped_lat, reshaped_lon,input_data))
        print(combined_data)





# Specify the output file
output_file = os.path.join(path_sh, 'output_csv', 'lat_lon_combined.csv')

# Create the output folder if it doesn't exist
os.makedirs(os.path.dirname(output_file), exist_ok=True)

# Save combined_data to CSV
with open(output_file, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Lat', 'Lon'])  # Add header
    writer.writerows(combined_data)

print(f"CSV file saved successfully: {output_file}")

