import numpy as np
import glob
import pandas as pd
import sys

def locate_latest_chains(abs_or_perc='',Errors='',N_samples='',Perc_true='',contamination_str='',photometric_str='',cosmo_type_str='',JAX=False,
                         cosmo_db_str='',return_db=False,list_of_file_indx = [],warmup=False,input_file=None,Prefix=None):
    if cosmo_db_str=='':
        cosmo_db_str=cosmo_type_str
    if input_file is None:
        file_search = f'./chains/{Prefix}{str(Errors).replace(".","p")}{abs_or_perc}_'+\
                                                    f'{N_samples}_samples_{Perc_true}_true_cosmo_{cosmo_db_str}.csv'+\
                                                    f'_ph_{photometric_str}_con_{contamination_str}'+\
                                                    f'_{cosmo_type_str}_'+\
                                                    'mcmc'*(JAX==False)+'JAX'*(JAX==True)+'_chains'
        print(file_search)
        mcmc_file_list = glob.glob(f'{file_search}*{"warmup"*warmup}.csv')
        if not warmup:
            mcmc_file_list = [elem for elem in mcmc_file_list if 'warmup' not in elem]
        print('File Found: ',mcmc_file_list)
    if input_file is not None:
        latest_chain_filename = input_file
        if not JAX: chains = np.load(latest_chain_filename) #[Walker,Step_Number,Parameter]
        if JAX: chains = pd.read_csv(latest_chain_filename)
        print(f'Loading mcmc chains from {latest_chain_filename}')
    elif len(list_of_file_indx)==0 and input_file is None:
        if not JAX: latest_chain_indx = np.argmax([float(elem.split('mcmc_chains_')[1].replace('.npy','')) \
                            for elem in mcmc_file_list])
        else: latest_chain_indx = np.argmax([float(elem.split('JAX_chains_')[1].replace('.csv','').replace('_warmup','')) \
                            for elem in mcmc_file_list])
        latest_chain_filename = mcmc_file_list[latest_chain_indx]
        if not JAX: chains = np.load(latest_chain_filename) #[Walker,Step_Number,Parameter]
        if JAX: chains = pd.read_csv(latest_chain_filename)
        print(f'Loading mcmc chains from {latest_chain_filename}')
    else:
        list_of_chains = [pd.read_csv(f'{file_search}_{elem}.csv') for elem in list_of_file_indx]
        for chain_i in list_of_chains:
            for chain_j in list_of_chains:
                assert chain_i.columns.tolist()==chain_j.columns.tolist() #Assert all columns are in each chain
        chains = pd.DataFrame()
        for column_i in list_of_chains[0].columns:
            column_i_str = column_i[:column_i.rfind('_')]
            for ii,chain_i in enumerate(list_of_chains):
                chains[f'{column_i_str}_{ii}'] = chain_i[column_i]
        latest_chain_filename = f'{file_search}_{list_of_file_indx[0]}.csv'
    if return_db:
        db_filename = latest_chain_filename.split('.csv')[0]+'.csv'
        db_filename = db_filename.replace('/chains','/databases').replace('SL_orig_','')
        db_i = pd.read_csv(db_filename)
        return chains,db_i
    else:
        return chains