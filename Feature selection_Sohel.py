#Feature selection
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
import numpy as np

def corr(predictions, targets):
    x = predictions
    y = targets
    x1 = np.array(x); y1 = np.array(y);
    x1_nanloc = np.isnan(x1)
    y1_nanloc = np.isnan(y1)
    nanloc = x1_nanloc + y1_nanloc    
    x1 = x1[nanloc==0].copy()
    y1 = y1[nanloc==0].copy()
        
    return np.corrcoef(x1, y1)[1][0]

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())


path = '/share/wildfire-2/sohel/Project/';
T_org_1 = pd.read_csv(path+"03_Sampling/Dataset_sj.csv")
input_name = ['SW38','SW38-IR87','SW38-IR96','SW38-IR105','SW38-IR112','SW38-IR123','SW38-IR133',
              'SW38_C','SW38-IR87_C','SW38-IR96_C','SW38-IR105_C','SW38-IR112_C','SW38-IR123_C','SW38-IR133_C','VI06','VI08','NR13','VI06_C','VI08_C','NR13_C']

T_org = T_org_1.loc[:,input_name].copy()

#--------------------------------------------------------------------------
# VIF 계산
max_vif = 9999

while max_vif>=5:
    if max_vif != 9999:
        T_org = T_org.drop([max_col], axis='columns')
    
    vif = pd.DataFrame()
    vif["VIF Factor"] = [variance_inflation_factor(T_org.values, i) for i in range(T_org.shape[1])]
    vif["features"] = T_org.columns
    vif
    
    vif['corr'] = [np.corrcoef(np.array(T_org.iloc[:,i]),np.array(T_org.iloc[:,-1]))[1][0] for i in range(T_org.shape[1])]
    
    max_col = T_org.columns[np.argmax(vif['VIF Factor'])]
    max_vif = vif['VIF Factor'][np.argmax(vif['VIF Factor'])]
    
    

print(vif)