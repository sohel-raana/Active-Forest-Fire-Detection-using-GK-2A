import os
import numpy as np
from scipy.io import loadmat, savemat
import pandas as pd
from datetime import timedelta

#paths
path_tj = '/share/wildfire-2/tjsung/Research/GK2A_ActiveFire_South_Korea/'
output_path = '/share/wildfire-2/sohel/Project/03_Sampling/Input/CNN'
path_sh = '/share/wildfire-2/sohel/Project/'

# Read grids
grid_path = os.path.join(path_sh, '01_Data/Grid')
# Load South Korea 2km data from MATLAB file
sko_2km = loadmat(os.path.join(grid_path, 'South_Korea_2km.mat'))['sko_2km']
# Load VGT 2km 2020 data from MATLAB file
vgt_2km = loadmat(os.path.join(grid_path, 'VGT_2km_2020.mat'))['vgt_2km'] * sko_2km

#Data table from previous vector extraction
path_data = '/share/wildfire-2/sohel/Project/03_Sampling/Dataset_sj.csv'
data_table = pd.read_csv(path_data)
data_table = data_table[(data_table['Isfire'] == 1)].copy()

# Function to extract the desired variables from _BT.mat and _ALB.mat files
def extract_data(x, y, bt_file, alb_file, vgt_2km, cld_data):
    
    bt_data = bt_file
    alb_data = alb_file
    
    # Vegetation Masking
    vgt_mask = np.isnan(vgt_2km)
    for var in ['sw38', 'ir87', 'ir96', 'ir105', 'ir112', 'ir123', 'ir133']:
        bt_data[var][vgt_mask] = np.nan

    for var in ['vi06', 'vi08', 'nr13']:
        alb_data[var][vgt_mask] = np.nan

    # Cloud Masking
    cld_mask = cld_data['cld'] < 2
    for var in ['sw38', 'ir87', 'ir96', 'ir105', 'ir112', 'ir123', 'ir133']:
        bt_data[var][cld_mask] = np.nan

    for var in ['vi06', 'vi08', 'nr13']:
        alb_data[var][cld_mask] = np.nan  

    
    # Extract variables from _BT.mat file
    bt_variables = bt_data['sw38'], bt_data['sw38'] - bt_data['ir87'], bt_data['sw38'] - bt_data['ir96'], \
                    bt_data['sw38'] - bt_data['ir105'], bt_data['sw38'] - bt_data['ir112'], \
                    bt_data['sw38'] - bt_data['ir123'], bt_data['sw38'] - bt_data['ir133']

    # Extract variables from _ALB.mat file
    alb_variables = alb_data['vi06'], alb_data['vi08'], alb_data['nr13']
    
    # Create a 15x15x10 array to store the extracted variables
    extracted_data = np.zeros((15, 15, 10))
    
    # Calculate the indices for the region centered at (x, y)
    start_x, start_y = max(0, x - 7), max(0, y - 7)
    end_x, end_y = start_x + 15, start_y + 15
    
    # Assign values to the array without handling NaN values yet
    for i, var in enumerate(bt_variables):
        extracted_data[:, :, i] = var[start_y:end_y, start_x:end_x]

    for i, var in enumerate(alb_variables):
        extracted_data[:, :, i + 7] = var[start_y:end_y, start_x:end_x]

    # Assign values to the array
    extracted_data[:, :, :7] = np.stack([var[start_y:end_y, start_x:end_x] for var in bt_variables], axis=-1)
    extracted_data[:, :, 7:] = np.stack([var[start_y:end_y, start_x:end_x] for var in alb_variables], axis=-1)

    # Now handle NaN values in the extracted_data
    for i in range(extracted_data.shape[2]):
        # Check the percentage of NaN values for each channel
        nan_percentage = np.isnan(extracted_data[:, :, i]).sum() / np.prod(extracted_data[:, :, i].shape)
        
        if nan_percentage > 0.75:
            print(f'Skipped sample because more than 75% of values are NaN in channel {i}')
            
            # Replace NaN values in the current channel with the median value
            channel_data = extracted_data[:, :, i]
            median_val = np.nanmedian(channel_data)
            nan_mask = np.isnan(channel_data)
            channel_data[nan_mask] = median_val
    
    return extracted_data

# Loop through rows in the data table
for index, row in data_table.iterrows():
    casenum = int(row['Casenum'])
    if casenum==100:
        # Extract components of the current time
        yr, mm, dd, hr, mn = str(int(row['Year'])).zfill(2), str(int(row['Month'])).zfill(2), str(int(row['Day'])).zfill(2), str(int(row['Hour'])).zfill(2), str(int(row['Minute'])).zfill(2)

        try:
            bt_data = loadmat(os.path.join(path_tj, f"01_Data/GK2A/Mat/{yr}/{mm}/{yr}{mm}{dd}_{hr}{mn}_BT.mat"))
            alb_data = loadmat(os.path.join(path_tj, f"01_Data/GK2A/Mat/{yr}/{mm}/{yr}{mm}{dd}_{hr}{mn}_ALB.mat"))
            cld_data = loadmat(os.path.join(path_tj, f"01_Data/GK2A/Mat/{yr}/{mm}/{yr}{mm}{dd}_{hr}{mn}_CLD.mat"))
        except FileNotFoundError:
            continue

        # Skip if GK-2A data is not real number
        if not np.isreal(np.dstack([bt_data, alb_data, cld_data])).all():
            continue
        
        x=int(row['X'])
        y=int(row['Y'])
        X=int(row['X'])-1
        Y=int(row['Y'])-1
        
        # Save the extracted data for the main pixel (X, Y)
        extracted_data = extract_data(X, Y, bt_data, alb_data, vgt_2km, cld_data)
            
        # Save the extracted data as a .npy file
        output_folder_npy = os.path.join(output_path,f'n_npy')
        os.makedirs(output_folder_npy, exist_ok=True)
        output_file_npy = os.path.join(output_folder_npy, f'{yr}{mm}{dd}_{hr}{mn}_{x}_{y}_cnn_input_{casenum}.npy')
        np.save(output_file_npy, extracted_data)
        print(f'Saved data at {output_file_npy}')
        
        # Extract and save 10 random samples where sko_2km values are not NaN
        random_sample_counter = 0
        sko_2km_not_nan_indices = np.argwhere(~np.isnan(sko_2km))
        for i in range(10):
            random_index = np.random.choice(len(sko_2km_not_nan_indices))
            random_sample = sko_2km_not_nan_indices[random_index]

            # Check if the random sample is within the central 15x15 area for case number 119 or 151
            center_row = int(row['Y']-1)
            center_col = int(row['X']-1)
            if casenum in [101,102] and center_row - 7 <= random_sample[1] <= center_row + 7 and center_col - 7 <= random_sample[0] <= center_col + 7:
                # Skip this random sample and generate a new one
                print(f'Skipped random sample {i + 1} within central 15x15 area for case {casenum}')
                continue

            ran_x = int(random_sample[0] + 1)
            ran_y = int(random_sample[1] + 1)

            # Extract the corresponding data for the random sample
            random_extracted_data = extract_data(random_sample[0], random_sample[1], bt_data, alb_data, vgt_2km, cld_data)
            
            # Save the extracted data for the random sample as .npy file
            output_folder_random_npy = os.path.join(output_path, f'N_random_npy')
            os.makedirs(output_folder_random_npy, exist_ok=True)
            random_output_file_npy = os.path.join(output_folder_random_npy, f'{yr}{mm}{dd}_{hr}{mn}_random_input_{casenum}_{i + 1}_{ran_x}_{ran_y}.npy')
            np.save(random_output_file_npy, random_extracted_data)
            print(f'Saved random sample {i + 1} at {random_output_file_npy}')
            
            random_sample_counter += 1
            if random_sample_counter == 10:
                break


print('All data Saved')