import pandas as pd
import glob
import numpy as np
import traceback
def run_general_check():
    db_files = glob.glob('./databases/*.csv')
    for file_i in db_files:
        db_i = pd.read_csv(file_i)
        try:
            if (db_i[db_i['spec']==1]['P_tau']>=0.99).all(): pass
            else: print(f'Failed spec P_tau check: {file_i}')
            mean_P_tau = np.round(db_i['P_tau'].mean(),2)
            lens_frac = np.round((db_i['FP_bool']==0).sum()/len(db_i),2)
            if abs(mean_P_tau-lens_frac)<0.1: pass
            else: print(f'Failed mean P_tau check: {file_i}',mean_P_tau,lens_frac)
        except:
            print('ERROR',print(file_i),db_i.columns)

def run_db_check(file_i):
    print(f'Checking {file_i}')
    class zBEAMS_Database_Error(Exception):
        pass
    db_i = pd.read_csv(file_i)
    try: db_i['spec'];db_i['P_tau'];db_i['FP_bool']
    except Exception as ex:print('ERROR',ex,print(file_i),db_i.columns);return
    if (db_i[db_i['spec']==1]['P_tau']>=0.99).all(): pass
    else: raise zBEAMS_Database_Error(f'Failed spec P_tau check: {file_i}');assert False
    mean_P_tau = np.round(db_i['P_tau'].mean(),2)
    lens_frac = np.round((db_i['FP_bool']==0).sum()/len(db_i),2)
    if abs(mean_P_tau-lens_frac)<0.1: pass
    else: raise zBEAMS_Database_Error(f'Failed mean P_tau check: {file_i},{mean_P_tau},{lens_frac}');assert False
