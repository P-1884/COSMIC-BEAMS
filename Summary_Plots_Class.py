import numpy as np
import pandas as pd
import matplotlib.pyplot as pl
from scipy.stats import gaussian_kde
from plot_JAX_chains import plot_JAX_chains
from retrieve_chain_files import retrieve_chain_files
from locate_latest_chains import locate_latest_chains
class summary_plots:
    def __init__(self,python_out_file):
        self.python_out_file = python_out_file
        # print('Reading .out file')
        self.warmup_JAX_chains_file = retrieve_chain_files(self.python_out_file,warmup=True)
        self.JAX_chains_file = retrieve_chain_files(self.python_out_file)
        print(f'Input db file: {retrieve_chain_files(self.python_out_file,database_file=True)}')
        self.input_db = pd.read_csv(retrieve_chain_files(self.python_out_file,database_file=True))
        # print('Finished reading .out file')
        if self.warmup_JAX_chains_file is not None:
            self.warmup_JAX_chains = locate_latest_chains(input_file = self.warmup_JAX_chains_file,JAX=True)
        else: self.warmup_JAX_chains = None
        if self.JAX_chains_file is not None:
            self.JAX_chains = locate_latest_chains(input_file = self.JAX_chains_file,JAX=True)
        else: self.JAX_chains = None
    def plot_chains(self):
        if self.warmup_JAX_chains is not None: 
            print('Plotting Warmup')
            plot_JAX_chains(self.warmup_JAX_chains,plot_hist=True,exclude_list = [])
        if self.JAX_chains is not None: 
            print('Plotting full chains')
            plot_JAX_chains(self.JAX_chains,plot_hist=True,exclude_list = [])
    def r_alpha_plots(self,N_chains,warmup_only=False,burnin=0,ylim = None,
                      fit_type='lognorm',fit_to='r_true'):
        if self.JAX_chains is not None and (not warmup_only): 
            print('Plotting Full Samples')
            r_alpha_investigation(self.input_db,self.JAX_chains.loc[burnin:].reset_index(drop=True),
                                  N_chains=N_chains,ylim=ylim,fit_type=fit_type,fit_to=fit_to)
        else: 
            print('Plotting from Warmup')
            r_alpha_investigation(self.input_db,self.warmup_JAX_chains.loc[burnin:].reset_index(drop=True),
                                N_chains=N_chains,ylim=ylim,fit_type=fit_type,fit_to=fit_to)
    def plot_extra_params(self,parameter_list,N_chains,plot_hist=True,warmup_only=False):
        title_dict = {'alpha_scale':'$\\alpha_\sigma$','alpha_s':'$\\alpha_s$','alpha_mu':'$\\alpha_\mu$'}
        if warmup_only: chains_to_plot = self.warmup_JAX_chains
        else: chains_to_plot = self.JAX_chains
        if plot_hist:
            fig,ax = pl.subplots(2,len(parameter_list),figsize=(5*len(parameter_list),10))
        else:
            fig,ax = pl.subplots(1,len(parameter_list),figsize=(5*len(parameter_list),5))
        for p_i,parameter in enumerate(parameter_list):
            for chain in range(N_chains):
                if plot_hist: 
                    ax_0 = ax[0,p_i];ax_1 = ax[1,p_i]
                else: ax_0 = ax[p_i]
                ax_0.plot(chains_to_plot[f'{parameter}_{chain}'],alpha=0.5)
                ax_0.set_xlabel('Steps')
            for chain in range(N_chains):
                if plot_hist: 
                    ax_0 = ax[0,p_i];ax_1 = ax[1,p_i]
                    hist_dict = {'bins':np.linspace(*ax_0.get_ylim(),30),'alpha':0.5,'edgecolor':'k'}
                    ax_1.hist(chains_to_plot[f'{parameter}_{chain}'],**hist_dict)
            try: ax_0.set_title(title_dict[parameter],fontsize=18,fontweight='bold')
            except: ax_0.set_title(parameter,fontsize=18,fontweight='bold')
        pl.tight_layout()
        pl.show()
    def plot_all(self,parameter_list,N_chains,plot_hist):
        self.plot_chains()
        # self.r_alpha_plots(N_chains,warmup_only=True,burnin=200,ylim=(0,5),
        #                    fit_type='truncnorm',fit_to='r_obs_contam')
        # self.plot_extra_params(parameter_list,N_chains,plot_hist,warmup_only=False)
        if self.JAX_chains_file is not None: return self.JAX_chains,self.input_db
        else: print('Returning WARMUP Chains');return self.warmup_JAX_chains,self.input_db