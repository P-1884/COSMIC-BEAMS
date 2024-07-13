print('Loading Packages')
import os
os.environ['JAX_PLATFORMS'] = 'cpu'
# os.environ['JAX_ENABLE_X64']='True'
# import jax
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
import glob
import sys
import jax
import os
try:importlib.reload(sys.modules['mcmcfunctions_SL'])
except Exception as ex: print(f'Cannot reload: {ex}')

from mcmcfunctions_SL import mcmc_SL

Om_fid = 0.3;Ode_fid = 0.7;H0_fid = 70;w_fid = -1.0;wa_fid=0

cosmo_type = 'wCDM'
'''
Have shown the JAX and emcee modules give answers in agreement for a very simple cosmology. Need to further
demonstrate this with more complex cosmologies (inc w0wa cosmology which emcee doesn't yet have?), but most
importantly including contamination + photometry.
'''

os.chdir('/mnt/users/hollowayp/zBEAMS/')
db_100k = pd.read_csv('./databases/real_paltas_population_TP_100000_FP_0_Spec_10000_P_1.0.csv')
db_100k_phot = pd.read_csv('./databases/real_paltas_population_TP_100000_FP_0_Spec_0_P_1.0.csv')
db_10k = pd.read_csv('./databases/real_paltas_population_TP_10000_FP_0_Spec_1000_P_1.0.csv')
db_10k_phot = pd.read_csv('/mnt/users/hollowayp/zBEAMS/databases/real_paltas_population_TP_10000_FP_0_Spec_0_P_1.0.csv')

from mcmcfunctions_SL_JAX import j_likelihood_SL,run_MCMC

class compare_likelihoods:
    def __init__(self,db_0,db_1=None):
        self.db_0 = db_0
        self.db_1 = db_1
    def generate_likelihood_dict(self,db,OM,Ok,w,wa):
        zL_obs_low_lim = np.array(db['zL_obs']-5*db['sigma_zL_obs'])
        zL_obs_low_lim = zL_obs_low_lim*(zL_obs_low_lim>0) #Minimum value is 0
        zL_obs_up_lim = np.array(db['zL_obs']+5*db['sigma_zL_obs'])
        zL = np.random.uniform(zL_obs_low_lim,high = zL_obs_up_lim)
        #
        zS_obs_low_lim = np.array(db['zS_obs']-5*db['sigma_zS_obs'])
        zS_obs_low_lim = zS_obs_low_lim*(zS_obs_low_lim>0)
        zS_obs_up_lim = np.array(db['zS_obs']+5*db['sigma_zS_obs'])
        zS = np.random.uniform(zL,high = zS_obs_up_lim)
        likelihood_dict = {'H0':70,'OM':OM, 'Ok':Ok, # The default is Om=0 and Ok = 0. This is so when one of them is updated
                            # in likelihood_dict below, it doesn't require Ode to be >1. Otherwise if I let Om or Ok vary between
                            # [0,1] and the other (Ok or Om) is non-zero by default, it would require Ode to be >1.
                            'w':w,'wa':wa,
                            'sigma_r_obs':db['sigma_r_obs'].to_numpy(),
                            'sigma_r_obs_2':1000*jnp.max(jnp.array(db['sigma_r_obs'])),
                            'zL':zL,
                            'zS':zS,
                            'zL_obs':db['zL_obs'].to_numpy(),'zS_obs':db['zS_obs'].to_numpy(),
                            'zL_sigma':db['sigma_zL_obs'].to_numpy(),'zS_sigma':db['sigma_zS_obs'].to_numpy(),
                            'r_obs':db['r_obs_contam'].to_numpy(),'sigma_r_obs':db['sigma_r_obs'].to_numpy(),
                            'mu_zL_g_L':np.nan,'mu_zS_g_L':np.nan,
                            'sigma_01_g_L':np.nan,'sigma_01_g_L':np.nan,
                            'P_tau':db['P_tau'].to_numpy()}
        return likelihood_dict

    def compare_likelihoods(self,OM,Ok,Ode,w,wa):
        self.L_dict_0 = self.generate_likelihood_dict(self.db_0,OM,Ok,w,wa)
        self.L_dict_1 = self.generate_likelihood_dict(self.db_1,OM,Ok,w,wa)
        # 
        model_args = {'cosmo_type':'wCDM',
                    'photometric':True,'contaminated':False,'H0':70,'key':None,
                    'likelihood_check':True,'cov_redshift':False,'early_return':False,
                    'batch_bool':False,'no_parent':True}
        for elem in ['zL_obs','zS_obs','sigma_zL_obs','sigma_zS_obs','r_obs','sigma_r_obs','sigma_r_obs_2','P_tau']:
            model_args[elem]=np.nan
        return j_likelihood_SL(**model_args,likelihood_dict=self.L_dict_0,spec_indx=np.where(self.db_0['spec'])[0])/\
               j_likelihood_SL(**model_args,likelihood_dict=self.L_dict_1,spec_indx=np.where(self.db_1['spec'])[0])

    def compare_likelihoods_for_contaminated_vs_not(self,OM,Ok,w,wa):
        self.L_dict_0 = self.generate_likelihood_dict(self.db_0,OM,Ok,w,wa)
        # 
        model_args = {'cosmo_type':'wCDM',
                    'photometric':True,'H0':70,'key':None,
                    'likelihood_check':True,'cov_redshift':False,'early_return':False,
                    'batch_bool':False,'no_parent':True}
        for elem in ['zL_obs','zS_obs','sigma_zL_obs','sigma_zS_obs','r_obs','sigma_r_obs','sigma_r_obs_2']:
            model_args[elem]=np.nan
        print({'Ignoring contamination':j_likelihood_SL(**model_args,contaminated=False,likelihood_dict=self.L_dict_0),
               'Including contamination':j_likelihood_SL(**model_args,contaminated=True,likelihood_dict=self.L_dict_0)})

import warnings

#def compare_contaminated_vs_not_likelihoods():

JAX_chains_that_worked = pd.read_csv('./chains/SL_orig_real_paltas_population_TP_10000_FP_0_Spec_1000_P_1.0.csv_ph_True_con_False_wCDM_JAX_chains_8_363.csv')

def compare_likelihoods_between_databases():
    fig,ax = pl.subplots(5,5,figsize=(25,25))
    range_dict = {'OM':(0,1),'Ok':(-1,1),'w':(-3,1),'wa':(-3,1),'Ode':(0,1)}
    ratio_list = []
    ratio_dict = {'OM':[],'Ok':[],'w':[],'wa':[],'Ode':[],'ratio':[]}
    with warnings.catch_warnings(action="ignore"):
        for i in tqdm(range(1000)):
            # param_dict = {elem:np.random.uniform(*range_dict[elem]) for elem in ['OM','Ode','w','wa']}
            # param_dict['Ok']=1-(param_dict['OM']+param_dict['Ode'])
            rand_chain = np.random.randint(0,3)
            rand_indx = np.random.randint(0,len(JAX_chains_that_worked))
            param_dict = {elem:JAX_chains_that_worked.loc[rand_indx][f'{elem}_{rand_chain}'] for elem in ['OM','Ode','Ok','w','wa']}
            ratio_i = compare_likelihoods(db_100k,db_10k).compare_likelihoods(**param_dict)
            for elem in ['OM','Ode','Ok','w','wa']: ratio_dict[elem].append(param_dict[elem])
            ratio_dict['ratio'].append(ratio_i)
            ratio_list.append(ratio_i)
            for p_0,param_0 in enumerate(['OM','Ode','w','wa','Ok']):
                for p_1,param_1 in enumerate(['OM','Ode','w','wa','Ok']):
                    ax[p_1,p_0].scatter(param_dict[param_0],param_dict[param_1],c=ratio_i,vmin=9.5,vmax=10.5)
                    ax[p_1,p_0].set_xlim(range_dict[param_0])
                    ax[p_1,p_0].set_ylim(range_dict[param_1])
                    if p_1==(len(ax)-1): ax[p_1,p_0].set_xlabel(param_0,fontsize=15)
                    if p_0==0: ax[p_1,p_0].set_ylabel(param_1,fontsize=15)
                    ax[p_1,p_0].tick_params(labelsize=12)
                    if p_1<p_0:
                        try: fig.delaxes(ax[p_1,p_0])
                        except: pass
    pd.DataFrame(ratio_dict).to_csv('./COSMIC_BEAMS_Likelihood_ratio_db.csv',index=False)
    pl.tight_layout()
    for fmt in ['.png','.pdf']: pl.savefig(f'./COSMIC_BEAMS_Likelihood_ratio{fmt}',bbox_inches='tight')
    pl.close()
    fig,ax = pl.subplots(figsize=(8,5))
    ax.hist(ratio_list,bins=np.linspace(9.5,10.5,50))
    print('Min,max',np.min(ratio_list),np.max(ratio_list))
    for fmt in ['.png','.pdf']: pl.savefig(f'./COSMIC_BEAMS_Likelihood_ratio_hist{fmt}',bbox_inches='tight')
    pl.close()
    for percentile in [1,25,50,75,99]:
        print(np.nanpercentile(ratio_list,percentile))

compare_likelihoods_between_databases()