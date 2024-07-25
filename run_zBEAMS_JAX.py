'''
Code description:
This code runs the main COSMIC-BEAMS inference. The input database (--filein) and other settings (e.g. whether the input systems are spectroscopic/photometric,
confirmed or impure) are added as optional arguments, before being passed to run_MCMC which is located in mcmcfunctions_SL_JAX.py. The outputs (MCMC chains) 
are then saved to a .csv file.
'''

#from tensorflow.python.client import device_lib
import argparse
import distutils
import time
import glob
import numpy as np

# Saves code each time it is run:
N_code_backups = len(glob.glob('./code_backups/mcmcfunctions_SL_JAX*'))
code_backup_file = f'./code_backups/mcmcfunctions_SL_JAX_{N_code_backups}_{np.round(time.time(),4)}.py'
print(f'Saving code backup to {code_backup_file}')
with open(code_backup_file,'w') as f:
    for line in open('./mcmcfunctions_SL_JAX.py'):
        f.write(line)

def argument_parser():
    parser = argparse.ArgumentParser()
    # Input database file
    parser.add_argument('--filein', type=str, help='Input file')
    # Whether the systems are photometric:
    parser.add_argument('--p', dest='p', type=lambda x:bool(distutils.util.strtobool(x)))
    # Whether the systems include contamination
    parser.add_argument('--c', dest='c', type=lambda x:bool(distutils.util.strtobool(x)))
    # What cosmology to assume (FlatLambdaCDM, LambdaCDM, wCDM, FlatwCDM)
    parser.add_argument('--cosmo', type=str, help='Cosmology type')
    # How many JAX steps to take in the MCMC:
    parser.add_argument('--num_samples', type=int, default = 1000, help='Number of samples')
    # How many JAX warmup steps to take in the MCMC:
    parser.add_argument('--num_warmup', type=int, default = 1000, help='Number of warmup samples')
    # How many chains in the MCMC:
    parser.add_argument('--num_chains', type=int, default = 2, help='Number of chains')
    # parser.add_argument('--N',type=int, default=10, help='Optional argument N')
    parser.add_argument('--target',type=float, default=0.8, help='Optional argument target_accept_prob')
    parser.add_argument('--cov_redshift', dest='cov_redshift', type=lambda x:bool(distutils.util.strtobool(x)),default=False)
    parser.add_argument('--batch', dest='batch', type=lambda x:bool(distutils.util.strtobool(x)),default=True)
    # Whether to keep wa fixed:
    parser.add_argument('--wa_const', dest='wa_const', type=lambda x:bool(distutils.util.strtobool(x)),default=False)
    # Whether to keep w0 fixed:
    parser.add_argument('--w0_const', dest='w0_const', type=lambda x:bool(distutils.util.strtobool(x)),default=False)
    # Random Seed:
    parser.add_argument('--key',type=int, default=0, help='Optional argument key_int')
    # Whether to use a gaussian mixture model for lens redshifts:
    parser.add_argument('--GMM_zL', action='store_true', help='Optional argument GMM_zL')
    # Whether to use a gaussian mixture model for source redshifts:
    parser.add_argument('--GMM_zS', action='store_true', help='Optional argument GMM_zS')
    parser.add_argument('--fixed_GMM', action='store_true', help='Optional argument fix_GMM')
    parser.add_argument('--nested',action='store_true',help='Optional argument nested sampling')
    parser.add_argument('--no_parent',action='store_true',help='Optional argument nested sampling')
    parser.add_argument('--initialise_to_truth',action='store_true',help='Optional argument initialise to true values')
    # Whether to use truncated gaussian distribution for lens redshifts:
    parser.add_argument('--trunc_zL',action='store_true',help='Optional argument truncate zL in the likelihood')
    # Whether to use truncated gaussian distribution for source redshifts:
    parser.add_argument('--trunc_zS',action='store_true',help='Optional argument truncate zS in the likelihood')
    parser.add_argument('--archive',action='store_true',help='Use archive version of likelihood function, from Github')
    parser.add_argument('--P_tau_dist',action='store_true',help='Use a distribution for P_tau')
    parser.add_argument('--sigma_P_tau',type=float, default=0.1, help='Sigma for P_tau distribution')
    parser.add_argument('--lognorm_parent',action='store_true',help='Use a lognormal distribution for the parent')
    parser.add_argument('--unimodal_beta', dest='unimodal_beta', type=lambda x:bool(distutils.util.strtobool(x)),default=True)
    parser.add_argument('--bimodal_beta', dest='bimodal_beta', type=lambda x:bool(distutils.util.strtobool(x)),default=False)
    args = parser.parse_args()
    return args

argv =  argument_parser()

filein = argv.filein
contaminated = argv.c
photometric = argv.p
cosmo_type = argv.cosmo
num_samples = argv.num_samples
num_warmup = argv.num_warmup
num_chains = argv.num_chains
target_accept_prob = argv.target
cov_redshift = argv.cov_redshift
batch_bool = argv.batch
wa_const = argv.wa_const
w0_const = argv.w0_const
key_int = argv.key
GMM_zL = argv.GMM_zL
GMM_zS = argv.GMM_zS
fixed_GMM = argv.fixed_GMM
nested = argv.nested
no_parent = argv.no_parent
init_to_truth = argv.initialise_to_truth
trunc_zL = argv.trunc_zL
trunc_zS = argv.trunc_zS
archive = argv.archive
P_tau_dist = argv.P_tau_dist
sigma_P_tau = argv.sigma_P_tau
lognorm_parent = argv.lognorm_parent
unimodal_beta = argv.unimodal_beta
bimodal_beta = argv.bimodal_beta
assert not (unimodal_beta and bimodal_beta) #Can't have both as True.
import sys

#os.environ["JAX_ENABLE_X64"] = 'True'
import numpyro
numpyro.enable_validation(True)
# numpyro.set_platform(platform='gpu')
import jax
import time
import os
from jax import local_device_count,default_backend,devices

from zbeamsfunctions_SL import likelihood_SL,likelihood_spec_contam_SL,likelihood_phot_contam_SL,likelihood_phot_SL,r_SL
from astropy.cosmology import LambdaCDM,FlatLambdaCDM,wCDM,FlatwCDM,w0waCDM
from zbeamsfunctions import mu_w,likelihood,likelihood_spec

# Uses previous version of code (for bug-finding purposes), should be False by default:
if archive: 
    try:
        from mcmcfunctions_SL_JAX_archive import j_likelihood_SL,run_MCMC
    except:
        print('FAILED TO FIND A GPU. DEFAULTING TO USING A CPU.')
        os.environ['JAX_PLATFORMS'] = 'cpu'
        from mcmcfunctions_SL_JAX_archive import j_likelihood_SL,run_MCMC
    print('RUNNING ARCHIVE VERSION')
else:
    try:
        from mcmcfunctions_SL_JAX import j_likelihood_SL,run_MCMC
    except:
        print('FAILED TO FIND A GPU. DEFAULTING TO USING A CPU.')
        os.environ['JAX_PLATFORMS'] = 'cpu'
        from mcmcfunctions_SL_JAX import j_likelihood_SL,run_MCMC

from Lenstronomy_Cosmology import Background, LensCosmo
from JAX_samples_to_dict import JAX_samples_to_dict
from mcmcfunctions import mcmc,mcmc_spec,mcmc_phot
from numpyro import distributions as dist, infer
from numpyro.infer import MCMC, NUTS, HMC
import matplotlib.patches as mpatches
from mcmcfunctions_SL import mcmc_SL
from scipy.stats import truncnorm
import matplotlib.lines as mlines
from cosmology_JAX import j_r_SL
from jax import random,grad,jit
import matplotlib.pyplot as pl
import jax.numpy as jnp
import jax_cosmo as jc
from tqdm import tqdm
import scipy.sparse
import pandas as pd
#import arviz as az
import numpy as np
import importlib
import corner
import emcee
import time
import glob
import sys
print('COMPS',local_device_count(),default_backend(),devices())

if fixed_GMM:
    GMM_zL_dict = {'mu_zL_g_L_A':0.3151891,
                   'mu_zL_g_L_B':0.62860335,
                   'sigma_zL_g_L_A':0.13988589,
                   'sigma_zL_g_L_B':0.24479157,
                   'w_zL':0.66072657}
    GMM_zS_dict = {'mu_zS_g_L_A':1.46348509,
                   'mu_zS_g_L_B':2.71829554,
                   'sigma_zS_g_L_A':0.54960578,
                   'sigma_zS_g_L_B':0.96609575,
                   'w_zS':0.67998525}
else:
    GMM_zL_dict = None
    GMM_zS_dict = None
print('ARGS',argv)
print('Filein',filein,'Contaminated',contaminated,'Photometric',photometric,'Cosmo',cosmo_type)
print('Num Samples',num_samples,'Num Warmup',num_warmup,'Num Chains',num_chains)
print('Target Accept Prob',target_accept_prob,'Cov Redshift',cov_redshift)
print('Batch Bool',batch_bool,'wa_const',wa_const,'w0_const',w0_const,'key_int',key_int)
print('GMM_zL',GMM_zL,'GMM_zS',GMM_zS,'fixed_GMM',fixed_GMM)
print('Nested Sampling',nested)
H0_fid = 70
db_in = pd.read_csv(filein)
print('DB In:',db_in)
print(db_in.columns)
numpyro.enable_x64(True)

# Use the true redshifts when not allowing for photometry, otherwise this biases the results as the error is already built into zL_/zS_obs:
if photometric:
    zL_to_use = jnp.array(db_in['zL_obs'])
    zS_to_use = jnp.array(db_in['zS_obs'])
else:
    zL_to_use = jnp.array(db_in['zL_true'])
    zS_to_use = jnp.array(db_in['zS_true'])

file_prefix = f'./chains/SL_orig_{filein.split("/")[-1]}'+\
              f'_ph_{photometric}_con_{contaminated}'+\
              f'_{cosmo_type}_JAX_chains'
file_search = glob.glob(f'{file_prefix}*')
N_chains_saved = len(file_search)
random_time = int(10*time.time())%1000
fileout = f'{file_prefix}_{N_chains_saved}_{random_time}.csv' #Will save first one as '_0'.
fileout_warmup = f'{file_prefix}_{N_chains_saved}_{random_time}_warmup.csv' #Will save first one as '_0'.
print(f'Will be saving file to: {fileout}')

if contaminated: assert (db_in['P_tau']!=1).all() #Otherwise this causes errors in the MCMC.

if archive:
    additional_args={}
else:
    additional_args = {'GMM_zL_dict':GMM_zL_dict,
                    'GMM_zS_dict':GMM_zS_dict,
                    'fixed_GMM':fixed_GMM,
                    'nested_sampling':nested,
                    'zL_true':db_in['zL_true'].to_numpy(),
                    'zS_true':db_in['zS_true'].to_numpy(),
                    'no_parent':no_parent,
                    'initialise_to_truth':init_to_truth,
                    'trunc_zL':trunc_zL,
                    'trunc_zS':trunc_zS,
                    'P_tau_dist':P_tau_dist,
                    'sigma_P_tau':sigma_P_tau,
                    'lognorm_parent':lognorm_parent,
                    'r_true':db_in['r_true'].to_numpy(),
                    'unimodal_beta':unimodal_beta,
                    'bimodal_beta':bimodal_beta}

sampler_S = run_MCMC(photometric = photometric,
                    contaminated = contaminated,
                    cosmo_type = cosmo_type,
                    zL_obs = zL_to_use,
                    zS_obs = zS_to_use,
                    sigma_zL_obs = jnp.array(db_in['sigma_zL_obs']),
                    sigma_zS_obs = jnp.array(db_in['sigma_zS_obs']),
                    r_obs = jnp.array(db_in['r_obs_contam']),
                    sigma_r_obs = jnp.array(db_in['sigma_r_obs']),
                    sigma_r_obs_2 = 10000*jnp.max(jnp.array(db_in['sigma_r_obs'])),
                    P_tau_0 = jnp.array(db_in['P_tau']),
                    num_warmup = num_warmup,
                    num_samples = num_samples,
                    num_chains = num_chains,
                    H0=H0_fid,
                    target_accept_prob=target_accept_prob,
                    cov_redshift=cov_redshift,
                    warmup_file=fileout_warmup,
                    batch_bool=batch_bool,
                    wa_const=wa_const,
                    w0_const=w0_const,
                    key_int=key_int,
                    GMM_zL=GMM_zL,
                    GMM_zS=GMM_zS,
                    **additional_args)

# Saves JAX chains to a pandas DataFrame:
a=JAX_samples_to_dict(sampler_S,separate_keys=True,cosmo_type=cosmo_type,wa_const=wa_const,w0_const=w0_const,fixed_GMM=fixed_GMM)
db_JAX = pd.DataFrame(a)
db_JAX.to_csv(fileout,index=False)
print('File search:',file_search)
print(f'Saving warmup to {fileout_warmup}')
print(f'Saving samples to {fileout}')
print('RANDOM OUTPUT SAMPLES:',set(np.random.choice((sampler_S.get_samples(True)['Ode']).flatten(),size=20)))

