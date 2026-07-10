'''
Code description:
This code runs the main COSMIC-BEAMS inference. The input database (--filein) and other settings (e.g. whether the input systems are spectroscopic/photometric,
confirmed or impure) are added as optional arguments, before being passed to run_MCMC_ which is located in mcmcfunctions_SL_JAX_margin_public.py. The outputs (MCMC chains) 
are then saved to a .csv file.
'''

from Lenstronomy_Cosmology import Background, LensCosmo
from mcmcfunctions import mcmc,mcmc_spec,mcmc_phot
from numpyro import distributions as dist, infer
from numpyro.infer import MCMC, NUTS, HMC
from database_checker import run_db_check
import matplotlib.patches as mpatches
from mcmcfunctions_SL import mcmc_SL
from scipy.stats import truncnorm
import matplotlib.lines as mlines
from cosmology_JAX_public import j_r_SL
from jax import random,grad,jit
import matplotlib.pyplot as pl
import jax.numpy as jnp
import jax_cosmo as jc
from tqdm import tqdm
import scipy.sparse
import pandas as pd
import numpy as np
import importlib
import distutils
import argparse
import numpyro
import corner
import emcee
import time
import glob
import time
import glob
import sys
import os
numpyro.enable_validation(True)
from jax import local_device_count,default_backend,devices
from astropy.cosmology import LambdaCDM,FlatLambdaCDM,wCDM,FlatwCDM,w0waCDM
from numpyro_truncnorm_GMM_fit_public import numpyro_truncnorm_GMM_fit

try:
    from mcmcfunctions_SL_JAX_margin_public import j_likelihood_SL,run_MCMC
except:
    print('FAILED TO FIND A GPU. DEFAULTING TO USING A CPU.')
    os.environ['JAX_PLATFORMS'] = 'cpu'
    from mcmcfunctions_SL_JAX_margin_public import j_likelihood_SL,run_MCMC

import jax

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
    # Whether to keep wa fixed:
    parser.add_argument('--wa_const', dest='wa_const', type=lambda x:bool(distutils.util.strtobool(x)),default=False)
    # Whether to keep w0 fixed:
    parser.add_argument('--w0_const', dest='w0_const', type=lambda x:bool(distutils.util.strtobool(x)),default=False)
    # Random Seed:
    parser.add_argument('--key',type=int, default=0, help='Optional argument key_int')
    # Whether to use a fixed zL-zS dependence in the likelihood:
    parser.add_argument('--fixed_zL_zS_dep',action='store_true',help='Use fixed zL zS dependence')
    # Whether to use fixed alpha, describing the r_obs distribution of the non-lenses (FP):
    parser.add_argument('--fixed_alpha',action='store_true',help='Use fixed alpha')
    # Whether to use fixed beta and gamma, describing the lens and "source" redshift distribution of the non-lenses:
    parser.add_argument('--fixed_beta_gamma',action='store_true',help='Use fixed beta and gamma distributions')
    # Whether to use the true zL and zS values in the likelihood (True) or not (False), rather than using the observed values (i.e. including measurement error)
    parser.add_argument('--use_true_z_phot_code',action='store_true',help='Use true z_phot code')
    # Whether to infer beta and gamma for the lens population:
    parser.add_argument('--beta_gamma_lens',action='store_true',help='Infer beta and gamma for lens population')
    # Whether to use a fixed beta and gamma distribution for the lens population:
    parser.add_argument('--fixed_beta_gamma_lens',action='store_true',help='Use fixed beta and gamma for lens population')
    # Whether to exclude the gamma_lens term from the likelihood function:
    parser.add_argument('--remove_gamma_lens',action='store_true',help='Remove gamma_lens term from likelihood function')
    # Whether to use the SLSim zL-zS dependence in the likelihood:
    parser.add_argument('--SLSim_zLzS_dep',action='store_true',help='Use SLSim zL zS dependence')
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
wa_const = argv.wa_const
w0_const = argv.w0_const
key_int = argv.key
fixed_alpha = argv.fixed_alpha
fixed_beta_gamma=argv.fixed_beta_gamma
use_true_z_phot_code = argv.use_true_z_phot_code
beta_gamma_lens = argv.beta_gamma_lens
fixed_beta_gamma_lens = argv.fixed_beta_gamma_lens
remove_gamma_lens = argv.remove_gamma_lens
fixed_zL_zS_dep = argv.fixed_zL_zS_dep
SLSim_zLzS_dep = argv.SLSim_zLzS_dep

print('COMPS',local_device_count(),default_backend(),devices())
print('ARGS',argv)

H0_fid = 70
db_in = pd.read_csv(filein)
numpyro.enable_x64(True)
num_warmp_hyp = 2000

if fixed_alpha:
    #If relevant, insert list of hyperparameters for the gaussian-mixture-model for the alpha distribution here - see numpyro_truncnorm_GMM_fit if required.
    alpha_dict = {'mu':[],
                  'scale':[],
                  'weights':jnp.array([])}
else:
    alpha_dict = {}

if fixed_beta_gamma:
    #If relevant, insert list of hyperparameters for the gaussian-mixture-model for the beta and gamma distribution here - see numpyro_truncnorm_GMM_fit if required.
    beta_dict = {'mu':[],
                  'scale':[],
                  'weights':jnp.array([])}
    gamma_dict = {'mu':[],
                  'scale':[],
                  'weights':jnp.array([])}
else:
    beta_dict = {}
    gamma_dict = {}

if fixed_beta_gamma_lens:
    #If relevant, insert list of hyperparameters for the gaussian-mixture-model for the beta_lens and gamma_lens distribution here - see numpyro_truncnorm_GMM_fit if required.
    beta_lens_dict = {'mu':[],
                  'scale':[],
                  'weights':jnp.array([])}
    gamma_lens_dict = {'mu':[],
                    'scale':[],
                    'weights':jnp.array([])}
else:
    beta_lens_dict = {}
    gamma_lens_dict = {}

if fixed_zL_zS_dep:
    if SLSim_zLzS_dep:
        s_dict = {'s_m': -0.11, 's_c': 0.545, 'scale_m': -0.261, 'scale_c': 2.514}
    else:
        #If relevant, insert the hyperparameters for the linear fit of the zS-zL dependence here.
        s_dict = dict(s_c = np.nan,
                    s_m = np.nan,
                    scale_c = np.nan,
                    scale_m = np.nan)
else: s_dict = None
   
# Use the true redshifts when not allowing for photometry, i.e. assume spectroscopy is perfect:
if photometric:
    zL_to_use = jnp.array(db_in['zL_obs'])
    zS_to_use = jnp.array(db_in['zS_obs'])
else:
    zL_to_use = jnp.array(db_in['zL_true'])
    zS_to_use = jnp.array(db_in['zS_true'])

additional_args = {
            'zL_true':jnp.array(db_in['zL_true']),
            'zS_true':jnp.array(db_in['zS_true']),
            'r_true':db_in['r_true'].to_numpy(),
            'fixed_beta_gamma':fixed_beta_gamma,
            'beta_dict':beta_dict,
            'gamma_dict':gamma_dict,
            'use_true_z_phot_code':use_true_z_phot_code,
            'beta_gamma_lens':beta_gamma_lens,
            'fixed_beta_gamma_lens':fixed_beta_gamma_lens,
            'beta_lens_dict':beta_lens_dict,
            'gamma_lens_dict':gamma_lens_dict,
            'fixed_alpha':fixed_alpha,
            'alpha_dict':alpha_dict,
            'remove_gamma_lens':remove_gamma_lens,
            's_dict':s_dict,
            'fixed_s':fixed_zL_zS_dep
}

sampler_S = run_MCMC(photometric = photometric,
                    contaminated = contaminated,
                    cosmo_type = cosmo_type,
                    zL_obs = zL_to_use,
                    zS_obs = zS_to_use,
                    sigma_zL_obs = jnp.array(db_in['sigma_zL_obs']),
                    sigma_zS_obs = jnp.array(db_in['sigma_zS_obs']),
                    r_obs = jnp.array(db_in['r_obs_contam']),
                    sigma_r_obs = jnp.array(db_in['sigma_r_obs']),
                    P_tau_0 = jnp.array(db_in['P_tau']),
                    num_warmup = num_warmup,
                    num_samples = num_samples,
                    num_chains = num_chains,
                    H0=H0_fid,
                    target_accept_prob=target_accept_prob,
                    wa_const=wa_const,
                    w0_const=w0_const,
                    key_int=key_int,
                    **additional_args)
