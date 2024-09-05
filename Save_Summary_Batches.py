print('Loading Packages')
import os
os.environ['JAX_PLATFORMS'] = 'cpu'
# os.environ['JAX_ENABLE_X64']='True'
import jax
# jax.config.update("jax_enable_x64", True)

from zbeamsfunctions_SL import likelihood_SL,likelihood_spec_contam_SL,likelihood_phot_contam_SL,likelihood_phot_SL,r_SL
from astropy.cosmology import LambdaCDM,FlatLambdaCDM,wCDM,FlatwCDM,w0waCDM
from zbeamsfunctions import mu_w,likelihood,likelihood_spec
from mcmcfunctions_SL_JAX import j_likelihood_SL,run_MCMC
from Lenstronomy_Cosmology import Background, LensCosmo
from scipy.stats import multivariate_normal as MVN
from sklearn.mixture import GaussianMixture as GMM
from mcmcfunctions import mcmc,mcmc_spec,mcmc_phot
from numpyro.infer import MCMC, NUTS, HMC, HMCECS
from numpyro import distributions as dist, infer
from squash_walkers import squash_walkers
from scipy.stats import truncnorm, norm
from numpyro.diagnostics import summary
import matplotlib.patches as mpatches
from mcmcfunctions_SL import mcmc_SL
import matplotlib.lines as mlines
from cosmology_JAX import j_r_SL
from jax import random,grad, jit
import matplotlib.pyplot as pl
from jax.random import PRNGKey
from importlib import reload
from subprocess import run
import jax.numpy as jnp
import jax_cosmo as jc
from tqdm import tqdm
import scipy.sparse
import pandas as pd
#import arviz as az #No longer compatible with scipy version 1.13.0 (previously used scipy version 1.11.4)
import numpy as np
import importlib
import numpyro
import corner
import emcee
import time
import glob
import sys
import jax
import os
from plot_JAX_chains import plot_JAX_chains
from retrieve_chain_files import retrieve_chain_files
from locate_latest_chains import locate_latest_chains
from Summary_Plots_Class import summary_plots
from plot_JAX_corner import plot_JAX_corner
try:importlib.reload(sys.modules['mcmcfunctions_SL'])
except Exception as ex: print(f'Cannot reload: {ex}')
from mcmcfunctions_SL import mcmc_SL
from numpyro_truncnorm_GMM_fit import numpyro_truncnorm_GMM_fit
from plot_JAX_corner import plot_JAX_corner,percentile_str,plot_mu_sig,range_dict,label_dict
from GMM_class import GMM_class
from test_kde_batching import kde_posterior_batching
from scipy.stats import gaussian_kde
import pickle

Om_fid = 0.3;Ode_fid = 0.7;H0_fid = 70;w_fid = -1.0;wa_fid=0
cosmo_type = 'wCDM'

from convert_ipynb_to_py import save_notebook_as_python_file 
def save_backup():
    # Saves code each time it is run:
    N_code_backups = np.max([len(glob.glob('./code_backups/mcmcfunctions_SL_JAX*')),
                            len(glob.glob('./code_backups/zBEAMS_Application_to_Strong_Lensing.py*'))])
    code_backup_time = np.round(time.time(),4)
    notebook_backup_file = f'./code_backups/zBEAMS_Application_to_Strong_Lensing_{N_code_backups}_{code_backup_time}.py'
    print(f'Saving notebook backup to {notebook_backup_file}')
    save_notebook_as_python_file('./zBEAMS_Application_to_Strong_Lensing.ipynb',notebook_backup_file)

save_backup()

def merge_chain_dict(chain_dict):
    for n_i,k_i in enumerate(chain_dict.keys()):
        for column_i in chain_dict[k_i].columns:
            old_column_i = column_i
            new_column_i = f'{column_i[:-2]}_{n_i}'
            chain_dict[k_i].rename({column_i:new_column_i},axis=1,inplace=True)
    JAX_chains = chain_dict[list(chain_dict.keys())[0]]
    for k_i in list(chain_dict.keys())[1:]:
        JAX_chains = pd.concat([JAX_chains,chain_dict[k_i]],axis=1)
    return JAX_chains

def combine_JAX_chains(python_out_file_chain_0,N_chain,exclude_chain_list = []):
    out_file_n0 = python_out_file_chain_0.split('-')[-1].replace('.out','')
    chain_n0 = python_out_file_chain_0.split('-')[-2].split('_')[-1]
    print(python_out_file_chain_0)
    print(out_file_n0,chain_n0,f'{chain_n0}_{out_file_n0}')
    prefix = python_out_file_chain_0.split(f'{chain_n0}-{out_file_n0}')[0]
    affix = python_out_file_chain_0.split(f'{chain_n0}-{out_file_n0}')[1]
    python_out_files = [f'{prefix}{i}-{int(out_file_n0)+i}{affix}' for i in list(set(np.arange(N_chain))-set(exclude_chain_list))]
    print(python_out_files)
    summary_plots_list = [summary_plots(elem) for elem in python_out_files]
    JAX_chains_list = {}
    for chain_ii,elem in tqdm(enumerate(summary_plots_list)):
        if elem.JAX_chains is not None:
            JAX_chains_list[chain_ii]=(elem.JAX_chains)
        else: 
            print(f'No sample chain found: returning warmup chains instead: {python_out_files[chain_ii]}')
            JAX_chains_list[chain_ii] = elem.warmup_JAX_chains
        if chain_ii==0: db_in_list = elem.input_db
    return merge_chain_dict(JAX_chains_list),db_in_list

try:
    argv = sys.argv
    print(argv)
    python_prefix = argv[1]
    python_out_n0 = int(argv[2])
    N_chains = int(argv[3])
    N_batch = int(argv[4])
    # python_out_files_chain_0 = [f'{python_prefix}_{i}_0-{python_out_n0+N_chains*i}.out' for i in range(N_batch)]
    # python_out_files_chain_0 = ['python3.11-delta_z_0.1_0_0-64021.out','python3.11-delta_z_0.1_1_0-64051.out',
    #                             'python3.11-delta_z_0.1_2_0-64081.out','python3.11-delta_z_0.1_3_0-64111.out',
    #                             'python3.11-delta_z_0.1_4_0-64141.out']
    # python_out_files_chain_0 = ['python3.11-delta_z_0.5_0_0-64031.out','python3.11-delta_z_0.5_1_0-64061.out',
    #                             'python3.11-delta_z_0.5_2_0-64091.out','python3.11-delta_z_0.5_3_0-64121.out',
    #                             'python3.11-delta_z_0.5_4_0-64151.out']
    # python_out_files_chain_0 = ['python3.11-delta_z_0.8_0_0-64041.out','python3.11-delta_z_0.8_1_0-64071.out',
    #                             'python3.11-delta_z_0.8_2_0-64101.out','python3.11-delta_z_0.8_3_0-64131.out',
    #                             'python3.11-delta_z_0.8_4_0-64161.out']
    python_out_files_chain_0 = ['python3.11-Subbatching_0_0-75609.out','python3.11-Subbatching_2_0-75689.out',
                                'python3.11-Subbatching_4_0-75769.out','python3.11-Subbatching_6_0-75849.out',
                                'python3.11-Subbatching_8_0-75930.out','python3.11-Subbatching_1_0-75649.out',
                                'python3.11-Subbatching_3_0-75729.out','python3.11-Subbatching_5_0-75809.out',
                                'python3.11-Subbatching_7_0-75889.out','python3.11-Subbatching_9_0-75970.out']
    save_combined_chains = eval(argv[5])
    run_summary_batch=True
    try:
        bw_factor = eval(argv[6])
    except: bw_factor = None
    try:
        outfile = argv[7]
    except: outfile = None
    print(f'Running {python_out_files_chain_0[0]} etc with {N_chains} chains and {N_batch} batches')
    print('bw factor:',bw_factor)
    print('outfile:',outfile)
except Exception as ex:
    print(ex)
    run_summary_batch=False
    pass

def drop_extra_columns(db,P_tau=True,zL=True,zS=True,Ok=False,alpha_weights_2=False):
    columns_to_drop = []
    if P_tau: columns_to_drop += list(db.filter(regex='P_tau'))
    if zL: columns_to_drop += list(db.filter(regex='zL'))
    if zS: columns_to_drop += list(db.filter(regex='zS'))
    if Ok: columns_to_drop += list(db.filter(regex='Ok'))
    if alpha_weights_2: columns_to_drop += list(db.filter(regex='alpha_weights_2'))
    return db[db.columns.drop(columns_to_drop)]


class summary_batch():
    def __init__(self,python_out_file_0=None,N_batch=None,
                 exclude_batch_list = [],python_out_files_chain_0 = None, N_chains = None,
                 exclude_chain_dict = None,drop_extra_columns_bool=True,python_out_files=None):
        if N_batch is None: N_batch = len(python_out_files_chain_0)
        self.N_batch = N_batch-len(exclude_batch_list)
        self.python_out_file_0 = python_out_file_0
        if python_out_files_chain_0 is None:
            if python_out_files is None:
                out_file_n0 = python_out_file_0.split('-')[-1].replace('.out','')
                prefix = python_out_file_0.split(out_file_n0)[0]
                affix = python_out_file_0.split(out_file_n0)[1]
                self.python_out_files = [f'{prefix}{int(out_file_n0)+i}{affix}' for i in list(set(np.arange(N_batch))-set(exclude_batch_list))]
            else: self.python_out_files = python_out_files
            # print(self.python_out_files)
            summary_plots_list = [summary_plots(elem) for elem in self.python_out_files]
            self.JAX_chains_list = []
            for b_ii,elem in enumerate(summary_plots_list):
                if elem.JAX_chains is not None:
                    self.JAX_chains_list.append(elem.JAX_chains)
                else: 
                    print(f'No sample chain found: returning warmup chains instead: {self.python_out_files[b_ii]}')
                    self.JAX_chains_list.append(elem.warmup_JAX_chains)
            self.db_in_list = [elem.input_db for elem in summary_plots_list]
        else:
            assert python_out_file_0 is None
            self.JAX_chains_list = [];self.db_in_list = []
            for batch_ii in tqdm(range(N_batch)):
                exclude_chain_list_i = exclude_chain_dict[batch_ii] if exclude_chain_dict is not None else []
                J_list_i,d_list_i = combine_JAX_chains(python_out_files_chain_0[batch_ii],N_chains,exclude_chain_list = exclude_chain_list_i)
                if drop_extra_columns_bool:
                    J_list_i = drop_extra_columns(J_list_i,Ok=True,alpha_weights_2=True)
                self.JAX_chains_list.append(J_list_i);self.db_in_list.append(d_list_i)
        # self.JAX_chains_list = [summary_plots(elem).JAX_chains for elem in self.python_out_files]
        self.population_hyperparameters = ['OM','Ode','w','wa']+\
                            [f'alpha_{mid}_{comp}' for mid in ['scale','mu'] for comp in range(3)]+\
                            [f'alpha_{mid}_{comp}' for mid in ['weights'] for comp in range(2)]+\
                            ['s_c', 's_m', 'scale_c', 'scale_m']
    def plot_JAX_chains(self,plot_single=False):
        color_list = pl.rcParams['axes.prop_cycle'].by_key()['color']
        if not plot_single: fig,ax = pl.subplots(1,5,figsize=(25,5));title=None
        for batch_i,JAX_chains_i in enumerate(self.JAX_chains_list):
            if plot_single: fig,ax = pl.subplots(1,5,figsize=(25,5));title=f'Batch: {batch_i}'
            if JAX_chains_i is not None:
                plot_JAX_chains(JAX_chains_i,color_0 = color_list[batch_i],fig=fig,ax=ax,title=title)
            else: print('No chains found')
            if plot_single: pl.show()
        if not plot_single: pl.show()
    def plot_alpha_corner(self,burnin = 0,burnout=None):
        columns_to_plot = squash_walkers(self.JAX_chains_list[0]).filter(like='alpha',axis=1).columns.tolist()
        color_list = pl.rcParams['axes.prop_cycle'].by_key()['color']
        self.fig = pl.figure(figsize=(5*len(columns_to_plot),5*len(columns_to_plot)))
        for batch_i in range(self.N_batch):
            corner.corner(squash_walkers(self.JAX_chains_list[batch_i].loc[burnin:burnout])[columns_to_plot],
                        labels=columns_to_plot,
                        fig=self.fig,
                        color=color_list[batch_i],
                        hist_kwargs={'density':True},
                        hist2d_kwargs={'label':'_nolegend_'},
                        label_kwargs={'fontsize':21},
                        plot_datapoints=False)
        for ax_i in self.fig.get_axes():
            ax_i.tick_params(labelsize=15)
        pl.tight_layout()
        pl.show()
    def plot_full_hyperparameter_corner(self,burnin = 0,burnout=None):
        columns_to_plot = self.population_hyperparameters
        color_list = pl.rcParams['axes.prop_cycle'].by_key()['color']
        self.fig = pl.figure(figsize=(5*len(columns_to_plot),5*len(columns_to_plot)))
        for batch_i in range(self.N_batch):
            corner.corner(squash_walkers(self.JAX_chains_list[batch_i].loc[burnin:burnout])[columns_to_plot],
                        labels=columns_to_plot,
                        fig=self.fig,
                        color=color_list[batch_i],
                        hist_kwargs={'density':True},
                        hist2d_kwargs={'label':'_nolegend_'},
                        label_kwargs={'fontsize':21},
                        plot_datapoints=False)
        for ax_i in self.fig.get_axes():
            ax_i.tick_params(labelsize=15)
        pl.tight_layout()
        pl.show()
    def plot_inferred_alpha(self,burnin=0,burnout=np.nan,binmax=2):
        hist_dict = {'bins':np.linspace(0,binmax,41),'density':True,'edgecolor':'k','alpha':0.5}
        color_list = pl.rcParams['axes.prop_cycle'].by_key()['color']
        N_comp = len(squash_walkers(self.JAX_chains_list[0]).filter(like='alpha_weight',axis=1).columns)
        for batch_i in range(self.N_batch):
            fig,ax = pl.subplots(figsize=(8,5))
            JAX_chains_i = squash_walkers(self.JAX_chains_list[batch_i])
            db_in_i = self.db_in_list[batch_i]
            color_i = color_list[batch_i]
            ax.hist(db_in_i['r_obs_contam'][db_in_i['FP_bool']==1],
                    label='$r_{obs}$ (FP Only)',color='darkred',**hist_dict)
            ax.hist(db_in_i['r_obs_contam'][db_in_i['FP_bool']==0],
                    label='$r_{obs}$ (TP Only)',color='darkgreen',**hist_dict)
            ax.hist(db_in_i['r_obs_contam'],label='$r_{obs}$ (All)',**hist_dict,fill=False)
            ax.legend([])
            GMM_truncnorm_best_fit = numpyro_truncnorm_GMM_fit(db_in_i['r_obs_contam'][db_in_i['FP_bool']==1].to_numpy(),
                                                            N_comp=3)
            for k_i in GMM_truncnorm_best_fit.keys():
                GMM_truncnorm_best_fit[k_i] = [float(GMM_truncnorm_best_fit[k_i][ii]) for ii in range(len(GMM_truncnorm_best_fit[k_i]))]
            GMM_class(**GMM_truncnorm_best_fit).plot(trunc_at_zero=True,X_plot=np.linspace(0,binmax,1001),ax=ax,
                                                    plot_components=False,total_color=color_i)
            burnout_i = np.nanmax([burnout,len(JAX_chains_i)-1])
            for ii in np.linspace(burnin,burnout_i,40,dtype=int):
                GMM_dict = {'list_of_mu':[JAX_chains_i[f'alpha_mu_{comp_i}'].loc[ii] for comp_i in range(N_comp)],
                    'list_of_sigma':[JAX_chains_i[f'alpha_scale_{comp_ii}'].loc[ii] for comp_ii in range(N_comp)],
                    'list_of_weights':[JAX_chains_i[f'alpha_weights_{comp_iii}'].loc[ii] for comp_iii in range(N_comp)]}
                GMM_class(**GMM_dict,
                    ).plot(trunc_at_zero=True,X_plot=np.linspace(0,binmax,1001),
                    label_components=False,ax = ax,alpha=0.1,label='_nolegend_',plot_components=False,
                    total_color=color_i)
            ax.set_xlabel('$r_{obs} = \\frac{c^2}{4\pi}\cdot\\frac{\\theta_E}{\\sigma_v^2}$',fontsize=15)
            ax.set_ylabel('Probability Density',fontsize=15)
            pl.tight_layout()
            pl.show()
    def plot_JAX_corner(self,combined_inference_db_list = None,burnin_combo = 0,burnin_single = 0,burnout_single=None):
        key_list = ['OM','Ode','w','wa','Ok']
        if combined_inference_db_list is not None: plot_combo = True
        else: plot_combo=False
        self.fig_corner,self.ax_corner = pl.subplots(len(key_list),len(key_list),
                                                     figsize=(2.2*len(key_list),2.2*len(key_list)))
        if plot_combo:
            combined_inference_flat = pd.concat({k_i:combined_inference_db_list[k_i].loc[burnin_combo:] \
                                                for k_i in combined_inference_db_list.keys()},
                                                ignore_index=True).reset_index(drop=True)
            if 'Ok' in key_list:
                combined_inference_flat['Ok'] = 1-(combined_inference_flat['OM']+combined_inference_flat['Ode'])
            corner.corner(combined_inference_flat[key_list],
                        fig=self.fig_corner,
                        hist_kwargs={'density':True},
                        hist2d_kwargs={'label':'_nolegend_'},
                        range=[range_dict[k_i] for k_i in key_list],
                        label_kwargs={'fontsize':21},
                        plot_datapoints=False)
        self.fig_corner,self.ax_corner = plot_JAX_corner(
            [squash_walkers(elem.loc[burnin_single:burnout_single]) for elem in self.JAX_chains_list],
            truth_dict = {'OM':0.3,'Ode':0.7,'w':-1,'wa':0,'Ok':1-(Om_fid+Ode_fid)},
            range_dict = range_dict,
            label_dict = label_dict,
            key_list=key_list, 
            plot_Ok=False,
            burnin=0, #BURNIN SHOULD BE 0 IF USING SQUASH-WALKERS?! AS Otherwise some of the burnin steps are put in the middle? 
            add_text=False,
            alpha_hist2d=(0.1*plot_combo)+(1-plot_combo),
            alpha_hist1d=(0.5*plot_combo)+(1-plot_combo),
            fig=self.fig_corner,ax=self.ax_corner)
        for ax_i in self.ax_corner.flatten():
                ax_i.tick_params(labelsize=15)
        # pl.tight_layout()
        pl.show()
    def redshift_bias(self,burnin=0,fractional_scatter=False):
        color_list = pl.rcParams['axes.prop_cycle'].by_key()['color']
        fig,ax = pl.subplots(1,3,figsize=(15,5))
        Bias_dict = {'zL':[],'zS':[]}
        for batch_i in range(self.N_batch):
            JAX_chains_i = self.JAX_chains_list[batch_i].loc[burnin:]
            db_in_i = self.db_in_list[batch_i]
            TP_indx = np.where(db_in_i['FP_bool']==0)[0]
            TP_indx = TP_indx[TP_indx<100] #Only select first 100, i.e. the ones which were saved to csv
            color_i = color_list[batch_i]
            True_zL = db_in_i['zL_true'].loc[0:99].to_numpy()[TP_indx] #Only saved first 100 values to csv
            True_zS = db_in_i['zS_true'].loc[0:99].to_numpy()[TP_indx]
            Uncertainty_zL_i = JAX_chains_i.filter(like='zL',axis=1).apply(lambda x: np.std(x))[TP_indx]
            Uncertainty_zS_i = JAX_chains_i.filter(like='zS',axis=1).apply(lambda x: np.std(x))[TP_indx]
            Mean_zL_i = JAX_chains_i.filter(like='zL',axis=1).apply(lambda x: np.mean(x))[TP_indx]
            Mean_zS_i = JAX_chains_i.filter(like='zS',axis=1).apply(lambda x: np.mean(x))[TP_indx]
            Error_zL_i = Mean_zL_i.to_numpy() - True_zL
            Error_zS_i = Mean_zS_i.to_numpy() - True_zS
            if fractional_scatter: ax[0].errorbar(True_zL,Error_zL_i/True_zL,yerr=Uncertainty_zL_i/True_zL,label=f'Batch {batch_i}',color='darkblue',fmt='.',alpha=0.5)
            if fractional_scatter: ax[0].errorbar(True_zS,Error_zS_i/True_zS,yerr=Uncertainty_zS_i/True_zS,label=f'Batch {batch_i}',color='darkred',fmt='.',alpha=0.5)
            if not fractional_scatter: ax[0].errorbar(True_zL,Mean_zL_i,yerr=Uncertainty_zL_i,label=f'Batch {batch_i}',color='darkblue',fmt='.',alpha=0.5)
            if not fractional_scatter:ax[0].errorbar(True_zS,Mean_zS_i,yerr=Uncertainty_zS_i,label=f'Batch {batch_i}',color='darkred',fmt='.',alpha=0.5)
            # max_frac_err = np.ceil(np.max([np.max(abs(Error_zL_i/Uncertainty_zL_i)),
            #                        np.max(abs(Error_zS_i/Uncertainty_zS_i))]))
            max_frac_err=5
            hist_dict = {'alpha':0.5,'bins':np.linspace(-max_frac_err,max_frac_err,10*max_frac_err+1),'density':True,'edgecolor':'k','label':f'Batch {batch_i}'}
            scaled_error_zL_i = Error_zL_i/Uncertainty_zL_i; scaled_error_zS_i = Error_zS_i/Uncertainty_zS_i
            ax[1].hist(scaled_error_zL_i,**hist_dict,color='darkblue')
            ax[2].hist(scaled_error_zS_i,**hist_dict,color='darkred')
            Bias_dict['zL'].append(np.round(np.median(scaled_error_zL_i),2))
            Bias_dict['zS'].append(np.round(np.median(scaled_error_zS_i),2))
        print('Median Redshift Bias ($\sigma$):',Bias_dict)
        # if fractional_scatter: ax[0].set_ylim(-0.03,0.03)
        ax[0].set_xlabel('$z_{True}$',fontsize=15)
        if fractional_scatter: ax[0].set_ylabel('Fractional Residual',fontsize=15)
        if not fractional_scatter: ax[0].set_ylabel('Inferred $z$',fontsize=15)
        ax[1].set_xlabel('Posterior $z_L$ Error ($\sigma$)',fontsize=15)
        ax[2].set_xlabel('Posterior $z_S$ Error ($\sigma$)',fontsize=15)
        [ax[p_i].set_ylabel('Probability Density',fontsize=15) for p_i in range(1,3)]
        [ax_i.legend() for ax_i in ax]
        [ax_i.tick_params(labelsize=12) for ax_i in ax]
        pl.suptitle('Redshift Inference',fontsize=21)
        pl.tight_layout()
        pl.show()
    def plot_P_tau_prior(self):
        fig,ax = pl.subplots()
        for batch_i in range(self.N_batch):
            P_tau_0_i = self.db_in_list[batch_i]['P_tau']
            TP_indx = np.where(self.db_in_list[batch_i]['FP_bool']==0)[0]
            FP_indx = np.where(self.db_in_list[batch_i]['FP_bool']==1)[0]
            ax.hist([P_tau_0_i[TP_indx],P_tau_0_i[FP_indx]],
                    bins=np.linspace(0,1,21),density=True,edgecolor='k',alpha=1/self.N_batch,
                    stacked=True,color = ['green','red'])
        ax.set_xlabel('$P_{\\tau,0}$',fontsize=12)
        ax.set_ylabel('Probability Density',fontsize=12)
        ax.set_title('Prior Lens Probability',fontsize=18)
        pl.tight_layout()
        pl.show()
    def P_tau_posterior_plots(self,burnin_single = 0,only_central_P = False,burnout_single=None,stacked=True):
        for batch_i in range(self.N_batch):
            JAX_chains_i = self.JAX_chains_list[batch_i]
            db_in_i = self.db_in_list[batch_i]
            errorbar_dict = {'fmt':'.','alpha':0.5,'linewidth':0.1}
            for sys_i in range(2000):
                pl.errorbar(sys_i,y = np.mean(JAX_chains_i.loc[burnin_single:burnout_single][f'P_tau_{sys_i}_0']),
                                yerr = np.std(JAX_chains_i.loc[burnin_single:burnout_single][f'P_tau_{sys_i}_0']),
                                color = {True:'red',False:'darkgreen'}[db_in_i['FP_bool'][sys_i]],
                                **errorbar_dict)
        pl.xlabel('System Index',fontsize=12)
        pl.ylabel('Posterior Probability',fontsize=12)
        pl.title('Posterior Probabilities ($\mu\pm\sigma$) from Impure Sample',fontsize=15)
        pl.tight_layout()
        pl.show()
        d_P_tau_list_TP = []
        d_P_tau_list_FP = []
        fig,ax_0 = pl.subplots()#figsize=(10,5),sharey=True)
        for batch_i in range(self.N_batch):
            JAX_chains_i = self.JAX_chains_list[batch_i]
            db_in_i = self.db_in_list[batch_i]
            for sys_i in range(2000):
                if only_central_P: 
                    if db_in_i['P_tau'][sys_i]<0.2 or db_in_i['P_tau'][sys_i]>0.8: continue
                d_P_tau_i = np.mean(JAX_chains_i.loc[burnin_single:burnout_single][f'P_tau_{sys_i}_0'])-db_in_i['P_tau'][sys_i]
                ax_0.errorbar(sys_i,d_P_tau_i,
                                yerr = np.std(JAX_chains_i.loc[burnin_single:burnout_single][f'P_tau_{sys_i}_0']),fmt='.',
                            color = {True:'red',False:'darkgreen'}[db_in_i['FP_bool'][sys_i]],
                            alpha=0.5,linewidth=0.1)
                if db_in_i['FP_bool'][sys_i]==1: d_P_tau_list_FP.append(d_P_tau_i)
                if db_in_i['FP_bool'][sys_i]==0: d_P_tau_list_TP.append(d_P_tau_i)
        ax_0.set_xlim(ax_0.set_xlim())
        ax_0.plot(ax_0.get_xlim(),[0]*2,'k--',zorder=10)
        ax_0.set_xlabel('System Index',fontsize=12)
        ax_0.set_ylabel('Change in Lens Probability',fontsize=12)
        ax_0.set_title('Change in Lens Probabilities ($\mu\pm\sigma$) from Impure Sample',fontsize=15)
        pl.tight_layout()
        pl.show()
        d_P_tau_list_TP = np.array(d_P_tau_list_TP)
        d_P_tau_list_FP = np.array(d_P_tau_list_FP)
        print('Median change for TPs',np.median(d_P_tau_list_TP))
        print('Median change for FPs',np.median(d_P_tau_list_FP))
        d_P_tau_list = np.array(d_P_tau_list_TP.tolist()+d_P_tau_list_FP.tolist())
        bin_extrem = np.max([abs(np.floor(np.min(d_P_tau_list*100))/100),
                            abs(np.ceil(np.max(d_P_tau_list*100))/100)]) #Rounding down to 2dp
        fig,ax = pl.subplots()
        if stacked:
            ax.hist([d_P_tau_list_TP,d_P_tau_list_FP],
                bins=np.arange(-bin_extrem,bin_extrem,0.01),density=True,edgecolor='k',alpha=0.5,
                stacked=stacked,color = ['green','red'])
        else:
            for p_i in range(2):
                ax.hist([d_P_tau_list_TP,d_P_tau_list_FP][p_i],
                bins=np.arange(-bin_extrem,bin_extrem,0.01),density=True,edgecolor='k',alpha=0.5,
                stacked=stacked,color = ['green','red'][p_i])
        ax.set_xlabel('Change in Lens Probability',fontsize=12)
        ax.set_ylabel('Probability Density',fontsize=12)
        ax.set_ylim(ax.get_ylim())
        ax.plot([0,0],ax.get_ylim(),'k--')
        pl.tight_layout()
        pl.show()
    def combined_inference(self,burnin=0,burnout=None,N_steps=50000):
        print('NOTE: This function is still undergoing testing')
        self.combined_inference_samples,combined_inference_kde = kde_posterior_batching(self.JAX_chains_list,
                            [['alpha']],bw_factor = bw_factor).find_kde_product([squash_walkers(elem.loc[burnin:burnout]) for elem in self.JAX_chains_list],
                                                            len(self.JAX_chains_list),
                                                            N_steps,
                                                            ['alpha'],
                                                            population_hyperparameters=self.population_hyperparameters)
        #Samples: N_steps, N_walkers, N_params
        self.combined_inference_samples = {walker_i:pd.DataFrame(self.combined_inference_samples.get_chain()[:,walker_i,:],
                                                                 columns=self.population_hyperparameters)\
                                            for walker_i in range(self.combined_inference_samples.get_chain().shape[1])}
        return self
    def save_files_to_calculate_P_tau_posterior(self):
        N_previous = len(glob.glob(f'./P_tau_posterior_files/{self.python_out_file_0}_pop_hyperparams*'))
        np.save(f'./P_tau_posterior_files/{self.python_out_file_0}_pop_hyperparams_V{N_previous}',self.population_hyperparameters)
        for k_i in range(len(self.JAX_chains_list)):
            self.JAX_chains_list[k_i].to_csv(f'./P_tau_posterior_files/{self.python_out_file_0}_JAX_chains_V{N_previous}_{k_i}.csv',index=False)
        for k_ii in self.combined_inference_samples.keys():
            self.combined_inference_samples[k_ii].to_csv(f'./P_tau_posterior_files/{self.python_out_file_0}_combined_inference_samples_V{N_previous}_{k_ii}.csv',index=False)

if run_summary_batch:
    if outfile is None:
        outfile = f'/mnt/extraspace/hollowayp/zBEAMS_data/class_instances/{python_out_files_chain_0[0]}_{N_batch}_{N_chains}_pickle.pkl'
    summary_batch_fiducial = summary_batch(None,python_out_files_chain_0 = python_out_files_chain_0,N_chains=N_chains)
    with open(outfile, 'wb') as file:
        pickle.dump(summary_batch_fiducial, file) 
    if save_combined_chains:
        summary_batch_fiducial = summary_batch_fiducial.combined_inference(burnin=500,burnout=1999)
        with open(outfile, 'wb') as file:
            pickle.dump(summary_batch_fiducial, file) 


def plot_JAX_corner_chain_specific(self,combined_inference_db_list = None,burnin_combo = 0,burnin_single = 0,burnout_single=None,
                                   chains_to_plot = []):
    key_list = ['OM','Ode','w','wa','Ok']
    if combined_inference_db_list is not None: plot_combo = True
    else: plot_combo=False
    self.fig_corner,self.ax_corner = pl.subplots(len(key_list),len(key_list),
                                                    figsize=(2.2*len(key_list),2.2*len(key_list)))
    if plot_combo:
        combined_inference_flat = pd.concat({k_i:combined_inference_db_list[k_i].loc[burnin_combo:] \
                                            for k_i in combined_inference_db_list.keys()},
                                            ignore_index=True).reset_index(drop=True)
        if 'Ok' in key_list:
            combined_inference_flat['Ok'] = 1-(combined_inference_flat['OM']+combined_inference_flat['Ode'])
        corner.corner(combined_inference_flat[key_list],
                    fig=self.fig_corner,
                    hist_kwargs={'density':True},
                    hist2d_kwargs={'label':'_nolegend_'},
                    range=[range_dict[k_i] for k_i in key_list],
                    label_kwargs={'fontsize':21},
                    plot_datapoints=False)
    chains_list = []
    for chain_i in chains_to_plot:
        chains_list.append(squash_walkers(self.JAX_chains_list[chain_i].loc[burnin_single:burnout_single]))
    self.fig_corner,self.ax_corner = plot_JAX_corner(
        chains_list,
        truth_dict = {'OM':0.3,'Ode':0.7,'w':-1,'wa':0,'Ok':1-(Om_fid+Ode_fid)},
        range_dict = range_dict,
        label_dict = label_dict,
        key_list=key_list, 
        plot_Ok=False,
        burnin=0, #BURNIN SHOULD BE 0 IF USING SQUASH-WALKERS?! AS Otherwise some of the burnin steps are put in the middle? 
        add_text=False,
        alpha_hist2d=(0.1*plot_combo)+(1-plot_combo),
        alpha_hist1d=(0.5*plot_combo)+(1-plot_combo),
        fig=self.fig_corner,ax=self.ax_corner)
    for ax_i in self.ax_corner.flatten():
            ax_i.tick_params(labelsize=15)
    # pl.tight_layout()
    pl.show()


def plot_inferred_alpha_smallbugfixed(self,burnin=0):
    hist_dict = {'bins':np.linspace(0,2,41),'density':True,'edgecolor':'k','alpha':0.5}
    color_list = pl.rcParams['axes.prop_cycle'].by_key()['color']
    N_comp = len(squash_walkers(self.JAX_chains_list[0]).filter(like='alpha_weight',axis=1).columns)
    for batch_i in range(self.N_batch):
        fig,ax = pl.subplots(figsize=(8,5))
        JAX_chains_i = squash_walkers(self.JAX_chains_list[batch_i])
        db_in_i = self.db_in_list[batch_i]
        color_i = color_list[batch_i]
        ax.hist(db_in_i['r_obs_contam'][db_in_i['FP_bool']==1],
                label='$r_{obs}$ (FP Only)',color='darkred',**hist_dict)
        ax.hist(db_in_i['r_obs_contam'],label='$r_{obs}$ (All)',**hist_dict)
        ax.legend([])
        GMM_truncnorm_best_fit = numpyro_truncnorm_GMM_fit(db_in_i['r_obs_contam'][db_in_i['FP_bool']==1].to_numpy(),
                                                        N_comp=3)
        for k_i in GMM_truncnorm_best_fit.keys():
            GMM_truncnorm_best_fit[k_i] = [float(GMM_truncnorm_best_fit[k_i][ii]) for ii in range(len(GMM_truncnorm_best_fit[k_i]))]
        GMM_class(**GMM_truncnorm_best_fit).plot(trunc_at_zero=True,X_plot=np.linspace(0,2,1001),ax=ax,
                                                plot_components=False,total_color=color_i)
        for ii in np.linspace(burnin,len(JAX_chains_i)-1,40,dtype=int):
            GMM_dict = {'list_of_mu':[JAX_chains_i[f'alpha_mu_{comp_i}'].loc[ii] for comp_i in range(N_comp)],
                'list_of_sigma':[JAX_chains_i[f'alpha_scale_{comp_ii}'].loc[ii] for comp_ii in range(N_comp)],
                'list_of_weights':[JAX_chains_i[f'alpha_weights_{comp_iii}'].loc[ii] for comp_iii in range(N_comp)]}
            GMM_class(**GMM_dict,
                ).plot(trunc_at_zero=True,X_plot=np.linspace(0,2,1001),
                label_components=False,ax = ax,alpha=0.1,label='_nolegend_',plot_components=False,
                total_color=color_i)
        ax.set_xlabel('$r_{obs} = \\frac{c^2}{4\pi}\cdot\\frac{\\theta_E}{\\sigma_v^2}$',fontsize=15)
        ax.set_ylabel('Probability Density',fontsize=15)
        pl.tight_layout()
        pl.show()