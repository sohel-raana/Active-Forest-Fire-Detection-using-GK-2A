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
#------------------------------

'''Saving the File with Lat and Lon as CSV'''

# Specify the paths
lat_lon_combined_path = '/share/wildfire-2/sohel/Project/output_csv/lat_lon_combined.csv'
input_folder_path = '/share/wildfire-2/sohel/Project/03_Sampling/result/Wholeimage_RF/Case_num_119/trial_Final'
output_folder_path = '/share/wildfire-2/sohel/Project/csv_latlon/119/Scheme2/new'

# Read the lat_lon_combined CSV file
lat_lon_combined = pd.read_csv(lat_lon_combined_path)
print(lat_lon_combined)
print("lat_lon_combined columns:", lat_lon_combined.columns)

# Ensure the output folder exists; if not, create it
if not os.path.exists(output_folder_path):
    os.makedirs(output_folder_path)

# Loop through each file in the input folder
for filename in os.listdir(input_folder_path):
    if filename.endswith('.csv') and 'scheme2' in filename:
        # Construct the full path of the input CSV file
        input_file_path = os.path.join(input_folder_path, filename)

        # Read the input CSV file
        input_data = pd.read_csv(input_file_path)
        print(f"{filename} columns:", input_data.columns)
        print(input_data)

        # Extract column name from the filename (8th to 27th index)
        column_name = filename[8:27]  # Adjust the index range accordingly
        print(column_name)
        
        # Reset index for both DataFrames
        lat_lon_combined_reset = lat_lon_combined.reset_index(drop=True)
        input_data_reset = input_data.reset_index(drop=True)
        
        # Create a temporary table with all values
        temp_table = pd.concat([lat_lon_combined_reset, input_data_reset], axis=1)

        # Filter rows based on the extracted column to keep only 1 values
        temp_table_filtered = temp_table[temp_table[column_name] == 1]

        # Construct the full path of the output CSV file with "scheme2" prefix
        output_file_path = os.path.join(output_folder_path, f"{filename}")

        # Save the filtered data to the output CSV file
        temp_table_filtered.to_csv(output_file_path, index=False)

        # Print information about the processed file
        print(f"Processed: {filename} -> Output: {output_file_path}")

print("Merging and saving completed.")

