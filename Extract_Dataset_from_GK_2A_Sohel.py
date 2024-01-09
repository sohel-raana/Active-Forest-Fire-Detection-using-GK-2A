import os
import numpy as np
import pandas as pd
from scipy.io import loadmat
from datetime import timedelta

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

# Load South Korea 2km data from MATLAB file
sko_2km = loadmat(os.path.join(grid_path, 'South_Korea_2km.mat'))['sko_2km']

# Load VGT 2km 2020 data from MATLAB file
vgt_2km = loadmat(os.path.join(grid_path, 'VGT_2km_2020.mat'))['vgt_2km'] * sko_2km

# Clear variables
del sko_2km



#----------------------------------------------------

# Change directory to the reference path
reference_path = os.path.join(path_sh, '02_Reference')
os.chdir(reference_path)

# Read fire data
data_f = pd.read_excel('KFS_201907_202207_f.xlsx')

# Extract unique metadata
metadata = data_f.iloc[:, :12].drop_duplicates()

# Create start time for fire events
date_cols = metadata.iloc[:, 2:7].astype(int).astype(str)
date_str = date_cols.apply(lambda x: ''.join(x), axis=1)
minutes_offset = metadata.iloc[:, 6] % 2
st_time_f = pd.to_datetime(date_str, format='%Y%m%d%H%M') + pd.to_timedelta(minutes_offset, unit='m')

# Create end time for fire events
date_cols = metadata.iloc[:, 7:12].astype(int).astype(str)
date_str = date_cols.apply(lambda x: ''.join(x), axis=1)
minutes_offset = metadata.iloc[:, 11] % 2
ed_time_f = pd.to_datetime(date_str, format='%Y%m%d%H%M') - pd.to_timedelta(minutes_offset, unit='m')

# Combine start and end times into a DataFrame
time_f = pd.concat([st_time_f, ed_time_f], axis=1)
time_f.columns = ['st_time_f', 'ed_time_f']

# Display the resulting start and end times for fire events
#print(time_f)

#------------------------------------------------------------------------
# Read data_p
data_p = pd.read_excel('KFS_201907_202207_p.xlsx')

# Create start time for data_p
st_time_p = pd.to_datetime(data_p.iloc[:, 1:6].astype(int).astype(str).agg(''.join, axis=1), format='%Y%m%d%H%M')

# Create end time for data_p
ed_time_p = pd.to_datetime(data_p.iloc[:, 6:11].astype(int).astype(str).agg(''.join, axis=1), format='%Y%m%d%H%M')

#-----------------------------------------------------------

from datetime import timedelta
for ii in range(metadata.shape[0]):
    # Extract samples only within 1 day after the ignition
    st_time = time_f['st_time_f'].iloc[ii]
    
    if time_f['ed_time_f'].iloc[ii] > st_time + timedelta(days=1):
        ed_time = st_time + timedelta(days=1)
    else:
        ed_time = time_f['ed_time_f'].iloc[ii]

    # Fire pixel
    data_f_case = data_f[data_f['Casenum'] == metadata['Casenum'].iloc[ii]]
    dmg_area = data_f_case[['X_mod', 'Y_mod']].values

    #print(ii)
    for now in pd.date_range(st_time, ed_time, freq='2T'):
        
        # Current time
        #print("Current time:", now)

        # Extract components of the current time
        yr, mm, dd, hr, mn = now.year, now.month, now.day, now.hour, now.minute
        
        # Adjust to KST (Korean Standard Time)
        kst = now + timedelta(hours=9)

        # Construct file paths
        # Read GK-2A data
        try:
            bt_data = loadmat(os.path.join(path_tj, f"01_Data/GK2A/Mat/{yr}/{mm:02d}/{yr}{mm:02d}{dd:02d}_{hr:02d}{mn:02d}_BT.mat"))
            alb_data = loadmat(os.path.join(path_tj, f"01_Data/GK2A/Mat/{yr}/{mm:02d}/{yr}{mm:02d}{dd:02d}_{hr:02d}{mn:02d}_ALB.mat"))
            cld_data = loadmat(os.path.join(path_tj, f"01_Data/GK2A/Mat/{yr}/{mm:02d}/{yr}{mm:02d}{dd:02d}_{hr:02d}{mn:02d}_CLD.mat"))
        except FileNotFoundError:
            continue
        print(bt_data)
        # Skip if GK-2A data is not real number
        if not np.isreal(np.dstack([bt_data, alb_data, cld_data])).all():
            continue
        
        # General information
        lat = lat_2km
        lon = lon_2km
        xx = np.tile(np.arange(1, 901), (900, 1))
        yy = np.tile(np.arange(1, 901), (900, 1)).T

        # CLD masking
        mask = cld < 2
        lat[mask] = np.nan
        lon[mask] = np.nan
        xx[mask] = np.nan
        yy[mask] = np.nan
        sw38[mask] = np.nan
        ir87[mask] = np.nan
        ir96[mask] = np.nan
        ir105[mask] = np.nan
        ir112[mask] = np.nan
        ir123[mask] = np.nan
        ir133[mask] = np.nan
        vi06[mask] = np.nan
        vi08[mask] = np.nan
        nr13[mask] = np.nan

        # VGT masking
        mask = np.isnan(vgt_2km)
        lat[mask] = np.nan
        lon[mask] = np.nan
        xx[mask] = np.nan
        yy[mask] = np.nan
        sw38[mask] = np.nan
        ir87[mask] = np.nan
        ir96[mask] = np.nan
        ir105[mask] = np.nan
        ir112[mask] = np.nan
        ir123[mask] = np.nan
        ir133[mask] = np.nan
        vi06[mask] = np.nan
        vi08[mask] = np.nan
        nr13[mask] = np.nan

        
        #OK this this
        
        
        # Fire mask
        data_p_case = data_p[(st_time_p <= now) & (now <= ed_time_p)]
        for jj in range(data_p_case.shape[0]):
            xx[data_p_case.iloc[jj, 0] - 7:data_p_case.iloc[jj, 0] + 8,
               data_p_case.iloc[jj, 1] - 7:data_p_case.iloc[jj, 1] + 8] = np.nan
            yy[data_p_case.iloc[jj, 0] - 7:data_p_case.iloc[jj, 0] + 8,
               data_p_case.iloc[jj, 1] - 7:data_p_case.iloc[jj, 1] + 8] = np.nan

        # Extract fire sample
        var_stack = np.dstack((sw38, sw38 - ir87, sw38 - ir96, sw38 - ir105, sw38 - ir112, sw38 - ir123, sw38 - ir133,
                               vi06, vi08, nr13))
        var_ent = np.full((914, 914, var_stack.shape[2]), np.nan)
        var_ent[7:907, 7:907, :] = var_stack
        nan_ent = np.any(np.isnan(var_ent), axis=2)
        pxid = (dmg_area[:, 1] - 1) * 900 + dmg_area[:, 0]
        fire_tmp = []
        
        for jj in range(pxid.shape[0]):
            if nan_ent[dmg_area[jj, 0] + 7, dmg_area[jj, 1] + 7]:
                continue

            for ww in range(3, 8):
                nan_win = np.copy(nan_ent[dmg_area[jj, 0] + 7 - ww:dmg_area[jj, 0] + 7 + ww,
                                 dmg_area[jj, 1] + 7 - ww:dmg_area[jj, 1] + 7 + ww])
                nan_win[ww:ww + 3, ww:ww + 3] = True

                if np.sum(~nan_win) / ((2 * ww + 1) ** 2) <= 0.25:
                    continue

                var_win = np.copy(var_ent[dmg_area[jj, 0] + 7 - ww:dmg_area[jj, 0] + 7 + ww,
                                 dmg_area[jj, 1] + 7 - ww:dmg_area[jj, 1] + 7 + ww, :])

                if np.all(var_win[ww, ww, [2, 4]] <= (np.quantile(var_win[:, :, [2, 4]], 0.75, axis=(0, 1)) +
                                                      3 * np.percentile(var_win[:, :, [2, 4]], 75, axis=(0, 1)))):
                    break

                theta = Solar_zenith(kst, lon[pxid[jj]], lat[pxid[jj]])
                contx_win = np.copy(var_win)
                contx_win[ww:ww + 3, ww:ww + 3, :] = np.nan
                contx_mean = np.nanmean(contx_win, axis=(0, 1))
                contx = var_win[ww, ww, :] - contx_mean

                # Extract input for CNN
                # You can add your own code here

                fire_tmp.append([metadata.iloc[ii, 0], metadata.iloc[ii, 1], yr, mm, dd, hr, mn, lat[pxid[jj]],
                                 lon[pxid[jj]], dmg_area[jj, 0], dmg_area[jj, 1], sw38[pxid[jj]],
                                 sw38[pxid[jj]] - ir87[pxid[jj]], sw38[pxid[jj]] - ir96[pxid[jj]],
                                 sw38[pxid[jj]] - ir105[pxid[jj]], sw38[pxid[jj]] - ir112[pxid[jj]],
                                 sw38[pxid[jj]] - ir123[pxid[jj]], sw38[pxid[jj]] - ir133[pxid[jj]],
                                 contx[0:7], vi06[pxid[jj]], vi08[pxid[jj]], nr13[pxid[jj]], contx[7:10], theta])

                break

        fire_tmp = pd.DataFrame(fire_tmp, columns=['Casenum', 'FireNum', 'Year', 'Month', 'Day', 'Hour', 'Minute',
                                                    'Latitude', 'Longitude', 'X', 'Y', 'SW38', 'SW38-IR87', 'SW38-IR96',
                                                    'SW38-IR105', 'SW38-IR112', 'SW38-IR123', 'SW38-IR133', 'Contx1',
                                                    'Contx2', 'Contx3', 'Contx4', 'Contx5', 'Contx6', 'Contx7', 'VI06',
                                                    'VI08', 'NR13', 'Contx8', 'Contx9', 'Contx10', 'SolarZenith'])
        fire = pd.concat([fire, fire_tmp])

        # Randomly sample 10 times of nonfire samples
        try:
            xy = resample(np.array(np.where(~np.isnan(xx))), n_samples=10 * fire_tmp.shape[0], replace=False)
        except ValueError:
            xy = resample(np.array(np.where(~np.isnan(xx))), n_samples=10 * fire_tmp.shape[0], replace=True)

        # Extract nonfire sample
        var_stack = np.dstack((sw38, sw38 - ir87, sw38 - ir96, sw38 - ir105, sw38 - ir112, sw38 - ir123, sw38 - ir133,
                               vi06, vi08, nr13))
        var_ent = np.full((914, 914, var_stack.shape[2]), np.nan)
        var_ent[7:907, 7:907, :] = var_stack
        nan_ent = np.isnan(np.mean(var_ent, axis=2))
        pxid = (xy[1, :] - 1) * 900 + xy[0, :]
        nonfire_tmp = []

        for jj in range(pxid.shape[0]):
            if nan_ent[xy[0, jj] + 7, xy[1, jj] + 7]:
                continue

            for ww in range(3, 8):
                nan_win = np.copy(nan_ent[xy[0, jj] + 7 - ww:xy[0, jj] + 7 + ww,
                                 xy[1, jj] + 7 - ww:xy[1, jj] + 7 + ww])
                nan_win[ww:ww + 3, ww:ww + 3] = True

                if np.sum(~nan_win) / ((2 * ww + 1) ** 2) <= 0.25:
                    continue

                var_win = np.copy(var_ent[xy[0, jj] + 7 - ww:xy[0, jj] + 7 + ww,
                                 xy[1, jj] + 7 - ww:xy[1, jj] + 7 + ww, :])

                theta = Solar_zenith(kst, lon[pxid[jj]], lat[pxid[jj]])
                contx_win = np.copy(var_win)
                contx_win[ww:ww + 3, ww:ww + 3, :] = np.nan
                contx_mean = np.nanmean(contx_win, axis=(0, 1))
                contx = var_win[ww, ww, :] - contx_mean

                # Extract input for CNN
                # You can add your own code here

                nonfire_tmp.append([metadata.iloc[ii, 0], metadata.iloc[ii, 1], yr, mm, dd, hr, mn, lat[pxid[jj]],
                                    lon[pxid[jj]], xy[1, jj], xy[0, jj], sw38[pxid[jj]],
                                    sw38[pxid[jj]] - ir87[pxid[jj]], sw38[pxid[jj]] - ir96[pxid[jj]],
                                    sw38[pxid[jj]] - ir105[pxid[jj]], sw38[pxid[jj]] - ir112[pxid[jj]],
                                    sw38[pxid[jj]] - ir123[pxid[jj]], sw38[pxid[jj]] - ir133[pxid[jj]],
                                    contx[0:7], vi06[pxid[jj]], vi08[pxid[jj]], nr13[pxid[jj]], contx[7:10], theta])

                break

        nonfire_tmp = pd.DataFrame(nonfire_tmp, columns=['Casenum', 'FireNum', 'Year', 'Month', 'Day', 'Hour', 'Minute',
                                                          'Latitude', 'Longitude', 'X', 'Y', 'SW38', 'SW38-IR87',
                                                          'SW38-IR96', 'SW38-IR105', 'SW38-IR112', 'SW38-IR123',
                                                          'SW38-IR133', 'Contx1', 'Contx2', 'Contx3', 'Contx4',
                                                          'Contx5', 'Contx6', 'Contx7', 'VI06', 'VI08', 'NR13',
                                                          'Contx8', 'Contx9', 'Contx10', 'SolarZenith'])
        nonfire = pd.concat([nonfire, nonfire_tmp])

print("Processing completed.")

#-----------------------------------------------------------------

import scipy.io

# Assuming fire and nonfire are NumPy arrays
fire = np.concatenate((np.ones((fire.shape[0], 1)), fire), axis=1)
nonfire = np.concatenate((np.zeros((nonfire.shape[0], 1)), nonfire), axis=1)

# Combine fire and nonfire arrays
dataset = np.concatenate((fire, nonfire), axis=0)

# Check if dataset is empty
if dataset.size == 0:
    raise ValueError("Empty dataset")

# Convert to Pandas DataFrame
header = ['Isfire', 'Casenum', 'Year', 'Month', 'Day', 'Hour', 'Minute', 'Lat', 'Lon', 'X', 'Y',
          'SW38', 'SW38-IR87', 'SW38-IR96', 'SW38-IR105', 'SW38-IR112', 'SW38-IR123', 'SW38-IR133',
          'SW38_C', 'SW38-IR87_C', 'SW38-IR96_C', 'SW38-IR105_C', 'SW38-IR112_C', 'SW38-IR123_C', 'SW38-IR133_C',
          'VI06', 'VI08', 'NR13', 'VI06_C', 'VI08_C', 'NR13_C', 'SOZ']

dataset = pd.DataFrame(dataset, columns=header)

# Convert 'Isfire' column to integer type
dataset['Isfire'] = dataset['Isfire'].astype(int)

# Check if dataset has duplicate rows
if dataset.duplicated().any():
    print("Warning: Dataset contains duplicate rows. Removing duplicates.")
    dataset = dataset.drop_duplicates()

# Save as CSV
csv_path = 'path/to/save/Dataset_sohel_trial.csv'
dataset.to_csv(csv_path, index=False)

# Save as MAT
mat_path = 'path/to/save/Dataset_sohel_trial.mat'
scipy.io.savemat(mat_path, {'dataset': dataset.to_dict(orient='list')})
