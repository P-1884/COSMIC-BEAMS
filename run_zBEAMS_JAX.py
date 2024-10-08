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
from convert_ipynb_to_py import save_notebook_as_python_file 
# import wandb
import initialise_wandb
import os
os.environ['WANDB_API_KEY'] #Checking key exists
# try:
#     wandb.init(
#         # set the wandb project where this run will be logged
#         project="My_test_project",
#         # track hyperparameters and run metadata
#         config={
#         "test_var":1})
# except:pass

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
    parser.add_argument('--batch_version',action='store_true',help='Use batch version of likelihood function')
    parser.add_argument('--block_version',action='store_true',help='Use block version of likelihood function')
    parser.add_argument('--N_batch',type=int, default=1, help='Optional argument - how many batches to use, when batch_version==True')
    parser.add_argument('--P_tau_dist',action='store_true',help='Use a distribution for P_tau')
    parser.add_argument('--sigma_P_tau',type=float, default=0.1, help='Sigma for P_tau distribution')
    parser.add_argument('--lognorm_parent',action='store_true',help='Use a lognormal distribution for the parent')
    parser.add_argument('--unimodal_beta', dest='unimodal_beta', type=lambda x:bool(distutils.util.strtobool(x)),default=True)
    parser.add_argument('--bimodal_beta', dest='bimodal_beta', type=lambda x:bool(distutils.util.strtobool(x)),default=False)
    parser.add_argument('--true_zL_zS_dep',action='store_true',help='Use true P(zL|zS) relation in the likelihood')
    parser.add_argument('--memory_check',action='store_true',help='Print help')
    parser.add_argument('--fixed_alpha',action='store_true',help='Use fixed alpha')
    parser.add_argument('--fixed_beta_gamma',action='store_true',help='Use fixed beta and gamma distributions')
    parser.add_argument('--P_tau_regular',action='store_true',help='Use regularisation for P_tau distribution')
    parser.add_argument('--P_tau_regular_factor',type=float, default=0.05, help='Regularisation factor for P_tau distribution')
    parser.add_argument('--likelihood_scale_factor',action='store_true',help='Use likelihood scale factor to make FP and TPs have likelihoods of similar magnitude')
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
true_zL_zS_dep = argv.true_zL_zS_dep
batch_version = argv.batch_version
N_batch = argv.N_batch
memory_check = argv.memory_check
block_version = argv.block_version
fixed_alpha = argv.fixed_alpha
P_tau_regular=argv.P_tau_regular
P_tau_regular_factor=argv.P_tau_regular_factor
fixed_beta_gamma=argv.fixed_beta_gamma
likelihood_scale_factor=argv.likelihood_scale_factor
assert not (unimodal_beta and bimodal_beta) #Can't have both as True.
import sys

# Saves code each time it is run:
N_code_backups = np.max([len(glob.glob('./code_backups/mcmcfunctions_SL_JAX*')),
                         len(glob.glob('./code_backups/zBEAMS_Application_to_Strong_Lensing.py*'))])
code_backup_time = np.round(time.time(),4)
if block_version:
    code_backup_file = f'./code_backups/mcmcfunctions_SL_blockMH_{N_code_backups}_{code_backup_time}.py'
    print(f'Saving code backup to {code_backup_file}')
    with open(code_backup_file,'w') as f:
        for line in open('./mcmcfunctions_SL_blockMH.py'):
            f.write(line)
elif batch_version:
    code_backup_file = f'./code_backups/mcmcfunctions_SL_JAX_batch{N_code_backups}_{code_backup_time}.py'
    print(f'Saving code backup to {code_backup_file}')
    with open(code_backup_file,'w') as f:
        for line in open('./mcmcfunctions_SL_JAX_batch.py'):
            f.write(line)
else:
    code_backup_file = f'./code_backups/mcmcfunctions_SL_JAX_{N_code_backups}_{code_backup_time}.py'
    print(f'Saving code backup to {code_backup_file}')
    with open(code_backup_file,'w') as f:
        for line in open('./mcmcfunctions_SL_JAX.py'):
            f.write(line)

notebook_backup_file = f'./code_backups/zBEAMS_Application_to_Strong_Lensing_{N_code_backups}_{code_backup_time}.py'
print(f'Saving notebook backup to {notebook_backup_file}')
save_notebook_as_python_file('./zBEAMS_Application_to_Strong_Lensing.ipynb',notebook_backup_file)



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
from numpyro_truncnorm_GMM_fit import numpyro_truncnorm_GMM_fit

# Uses previous version of code (for bug-finding purposes), should be False by default:
if archive: 
    try:
        from mcmcfunctions_SL_JAX_archive import j_likelihood_SL,run_MCMC
    except:
        print('FAILED TO FIND A GPU. DEFAULTING TO USING A CPU.')
        os.environ['JAX_PLATFORMS'] = 'cpu'
        from mcmcfunctions_SL_JAX_archive import j_likelihood_SL,run_MCMC
    print('RUNNING ARCHIVE VERSION')
if batch_version:
    try:
        from mcmcfunctions_SL_JAX_batch import j_likelihood_SL,run_MCMC
    except:
        print('FAILED TO FIND A GPU. DEFAULTING TO USING A CPU.')
        os.environ['JAX_PLATFORMS'] = 'cpu'
        from mcmcfunctions_SL_JAX_batch import j_likelihood_SL,run_MCMC
    print('RUNNING BATCH VERSION')
elif block_version:
    try:
        from mcmcfunctions_SL_blockMH import j_likelihood_SL,run_MCMC
    except:
        print('FAILED TO FIND A GPU. DEFAULTING TO USING A CPU.')
        os.environ['JAX_PLATFORMS'] = 'cpu'
        from mcmcfunctions_SL_blockMH import j_likelihood_SL,run_MCMC
    print('RUNNING BLOCK VERSION')
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
from database_checker import run_db_check
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

run_db_check(filein)

if fixed_alpha:
    #Values previously based on /mnt/users/hollowayp/zBEAMS/databases/real_paltas_population_TP_100000_FP_100000_Spec_10000_P_0.5.csv
    # alpha_dict = {'mu':[0.6299770474433899, 0.07905451953411102, 0.34039413928985596],
    #               'scale':[0.4377945363521576,1.2255791425704956,0.23973476886749268],
    #               'weights':jnp.array([0.42739158868789673,0.09350378066301346,0.4791046380996704])}
    # alpha_dict = {'mu':[100,100,100],
    #               'scale':[0.1,0.1,0.1],
    #               'weights':jnp.array([0.1,0.1,0.8])}
    # print('USING PRESET WILD ALPHA DICT')
    FP_db = db_in[db_in['FP_bool']==1].copy().reset_index(drop=True)
    alpha_dist_fit = numpyro_truncnorm_GMM_fit(FP_db['r_obs_contam'].to_numpy(),N_comp=3)
    alpha_dict = {'mu':alpha_dist_fit['list_of_mu'],
                  'scale':alpha_dist_fit['list_of_sigma'],
                  'weights':jnp.array(alpha_dist_fit['list_of_weights'])}
else:
    alpha_dict = {}

print('fixed_alpha',fixed_alpha,'alpha dict',alpha_dict)

if fixed_beta_gamma:
    FP_db = db_in[db_in['FP_bool']==1].copy().reset_index(drop=True)
    beta_dist_fit = numpyro_truncnorm_GMM_fit(FP_db['zL_obs'].to_numpy(),N_comp=1)
    gamma_dist_fit = numpyro_truncnorm_GMM_fit(FP_db['zS_obs'].to_numpy(),N_comp=1)
    # beta_dict = {'mu':10,'scale':0.1,'weights':1}
    # gamma_dict = {'mu':10,'scale':0.1,'weights':1}
    # print('USING PRESET WILD BETA GAMMA DICT')
    beta_dict = {'mu':beta_dist_fit['list_of_mu'][0],
                'scale':beta_dist_fit['list_of_sigma'][0],
                'weights':beta_dist_fit['list_of_weights'][0]}
    gamma_dict = {'mu':gamma_dist_fit['list_of_mu'][0],
                'scale':gamma_dist_fit['list_of_sigma'][0],
                'weights':gamma_dist_fit['list_of_weights'][0]}
else:
    beta_dict = {}
    gamma_dict = {}

print('fixed_beta_gamma',fixed_beta_gamma,'beta dict',beta_dict,'gamma dict',gamma_dict)

# Use the true redshifts when not allowing for photometry, otherwise this biases the results as the error is already built into zL_/zS_obs:
if photometric:
    zL_to_use = jnp.array(db_in['zL_obs'])
    zS_to_use = jnp.array(db_in['zS_obs'])
else:
    zL_to_use = jnp.array(db_in['zL_true'])
    zS_to_use = jnp.array(db_in['zS_true'])

file_prefix = f'/mnt/extraspace/hollowayp/zBEAMS_data/chains/SL_orig_{filein.split("/")[-1]}'+\
              f'_ph_{photometric}_con_{contaminated}'+\
              f'_{cosmo_type}_JAX_chains'
file_search = glob.glob(f'{file_prefix}*')
N_chains_saved = len(file_search)
random_time = str(time.time())[-5:]
fileout = f'{file_prefix}_{N_chains_saved}_{key_int}_{random_time}.csv' #Will save first one as '_0'.
fileout_warmup = f'{file_prefix}_{N_chains_saved}_{key_int}_{random_time}_warmup.csv' #Will save first one as '_0'.
print(f'Will be saving file to: {fileout}')

# if contaminated: assert (db_in['P_tau']!=1).all() #Otherwise this causes errors in the MCMC.

additional_args = {}
if not archive:
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
                    'bimodal_beta':bimodal_beta,
                    'true_zL_zS_dep':true_zL_zS_dep,
                    'fixed_alpha':fixed_alpha,
                    'alpha_dict':alpha_dict,
                    'fixed_beta_gamma':fixed_beta_gamma,
                    'beta_dict':beta_dict,
                    'gamma_dict':gamma_dict,
                    'P_tau_regular':P_tau_regular,
                    'P_tau_regular_factor':P_tau_regular_factor,
                    'likelihood_scale_factor':likelihood_scale_factor}
if batch_version:
    additional_args['N_batch'] = N_batch

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

if batch_version:
    sampler_S,N_sys_per_batch = sampler_S
else: N_sys_per_batch = []
# Saves JAX chains to a pandas DataFrame:
a=JAX_samples_to_dict(sampler_S,separate_keys=True,cosmo_type=cosmo_type,wa_const=wa_const,w0_const=w0_const,fixed_GMM=fixed_GMM,
                      N_sys_per_batch = N_sys_per_batch)
db_JAX = pd.DataFrame(a)
db_JAX.to_csv(fileout,index=False)
print('File search:',file_search)
print(f'Saving warmup to {fileout_warmup}')
print(f'Saving samples to {fileout}')
print('RANDOM OUTPUT SAMPLES:',set(np.random.choice((sampler_S.get_samples(True)['Ode']).flatten(),size=20)))

