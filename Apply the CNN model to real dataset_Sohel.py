#Apply the CNN model to real dataset

import os
import numpy as np
from scipy.io import loadmat, savemat
import pandas as pd
from datetime import timedelta
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
#paths
path_tj = '/share/wildfire-2/tjsung/Research/GK2A_ActiveFire_South_Korea/'
output_path = '/share/wildfire-2/sohel/Project/03_Sampling/Input/CNN/result/CNN_sl'
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
    
    # Assign values to the array
    extracted_data[:, :, :7] = np.stack([var[start_y:end_y, start_x:end_x] for var in bt_variables], axis=-1)
    extracted_data[:, :, 7:] = np.stack([var[start_y:end_y, start_x:end_x] for var in alb_variables], axis=-1)

    # Now handle NaN values in the extracted_data
    for i in range(extracted_data.shape[2]):
        channel_data = extracted_data[:, :, i]
        median_val = np.nanmedian(channel_data)
        nan_mask = np.isnan(channel_data)
        channel_data[nan_mask] = median_val
    
    return extracted_data

all_fire_predictions = []

# Create the MinMaxScaler object outside the loop
scaler = MinMaxScaler()

# Loop through rows in the data table
for index, row in data_table.iterrows():
    casenum = int(row['Casenum'])
    if casenum == 119:
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

        X = int(row['X']) - 1
        Y = int(row['Y']) - 1

        # Save the extracted data for the main pixel (X, Y)
        extracted_data = extract_data(X, Y, bt_data, alb_data, vgt_2km, cld_data)

        # Fit the scaler to the extracted data
        scaler.fit(extracted_data.reshape(-1, 15 * 15 * 10))

        model_path = '/home/sohel/.vscode-server/data/User/best_model.h5'  # Replace with the actual path to your model file
        model = load_model(model_path)

        # Prepare the data for the model
        img_rows, img_cols, channels = 15, 15, 10
        input_data = extracted_data.reshape(1, img_rows, img_cols, channels)

        # Normalize the data using the fitted MinMaxScaler
        input_data = scaler.transform(input_data.reshape(-1, img_rows * img_cols * channels)).reshape(1, img_rows, img_cols, channels)

        # Predict fire/non-fire labels using the loaded model
        predictions = model.predict(input_data)
        f_predictions = predictions[:, 1]

        # Threshold predictions to obtain binary results
        fire_predictions = np.where(f_predictions > 0.5, 1, 0)
        
        # Create a 15x15x10 array to store the extracted variables
        visual = np.zeros((15, 15, 1))
        visual[:, :, :] = np.stack(fire_predictions, axis=-1)
        
        channel_index=0
        # Select the data for the chosen channel
        channel_data = visual[:, :, channel_index]
        
        plt.imshow(channel_data, cmap='grey')
        plt.colorbar()  # Add a colorbar for reference
        plt.show()



# Visualize the fire predictions as an image
plt.imshow(fire_predictions_2d, cmap='hot', interpolation='nearest', vmin=0, vmax=1)
plt.title('Detected Fire Areas')
plt.colorbar()
plt.show()
        
# Create an animation showing detected fire areas over the loop
def update(frame):
    ax.clear()
    ax.set_title(f'Frame {frame}')
    ax.imshow(all_fire_predictions[frame], cmap='hot', interpolation='nearest', vmin=0, vmax=1)
    ax.axis('off')

fig, ax = plt.subplots()
animation = FuncAnimation(fig, update, frames=len(data_table), interval=200, repeat=False)

# Save the animation as a GIF
animation.save('fire_detection_loop.gif', writer='imagemagick')

# Show the plot (optional)
plt.show()