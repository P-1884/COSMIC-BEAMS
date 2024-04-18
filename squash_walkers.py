import pandas as pd
import numpy as np

def squash_walkers(db,exclude=[],reshape=False,verbose=True):
    column_list = db.columns
    #rfind finds the index of the last occurrence of the character in a string: 
    column_set = list(set([elem[:elem.rfind('_')] for elem in column_list]))
    column_dict = {}
    for c_i in column_list:
        c_i_set = c_i[:c_i.rfind('_')]
        if int(c_i.split('_')[-1]) in exclude:
            if verbose: print(f'Excluding {c_i}')
            continue
        try:
            if reshape: column_dict[c_i_set].append((db[c_i]).tolist())
            else: column_dict[c_i_set].extend(np.array(db[c_i]))
        except:
            if reshape: column_dict[c_i_set] = [db[c_i].tolist()]
            else: column_dict[c_i_set] = list(db[c_i])
    if reshape: 
        for k_i in column_dict.keys():
            column_dict[k_i]=np.array(column_dict[k_i])
        return column_dict
    else: return pd.DataFrame(column_dict)