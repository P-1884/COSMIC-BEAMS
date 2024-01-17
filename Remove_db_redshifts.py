import sys
import pandas as pd
from tqdm import tqdm
argv = sys.argv
#argv = [1,'./chains/SL_orig_spectroscopic_db_1perc_1024_samples_80_true_cosmo_wCDM.csv_ph_True_con_True_wCDM_JAX_chains_1705008215.4188426.csv']
print('args:',argv)
filename = argv[1]
db = pd.read_csv(filename)
columns_to_save = []
for c_i in tqdm(db.columns):
    if (c_i[0:2] in ['zL','zS']) and (int(c_i.split('_')[1])>10):
        continue
    else:
        columns_to_save.append(c_i)

print('Saving the following columns:',columns_to_save)
db[columns_to_save].to_csv(filename,index_label=False)
