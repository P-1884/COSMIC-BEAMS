#from tensorflow.python.client import device_lib
import sys
argv = sys.argv
print('ARGS',argv)
#import os
#os.environ["JAX_ENABLE_X64"] = 'True'
import numpyro
import jax
import time
from jax import local_device_count,default_backend,devices
# if default_backend()=='cpu': => Doesn't seem to recognise more than one device - always just prints jax.devices=1.
#     numpyro.util.set_host_device_count(4)
#     print('Device count:', len(jax.devices()))

# def get_available_gpus():
#     local_device_protos = device_lib.list_local_devices()
#     return [x.name for x in local_device_protos if x.device_type == 'GPU']
# print('GPU',get_available_gpus())
# numpyro.set_platform(platform='gpu')
from zbeamsfunctions_SL import likelihood_SL,likelihood_spec_contam_SL,likelihood_phot_contam_SL,likelihood_phot_SL,r_SL
from astropy.cosmology import LambdaCDM,FlatLambdaCDM,wCDM,FlatwCDM,w0waCDM
from zbeamsfunctions import mu_w,likelihood,likelihood_spec
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
filein = argv[1]
contaminated = eval(argv[2])
photometric = eval(argv[3])
print('Contaminated',contaminated,type(contaminated),'Photometric',photometric,type(photometric))
print('Filein',filein)
cosmo_type = argv[4]
try:
    num_samples = int(argv[5])
    num_warmup = np.max([100,num_samples//10])
except:
    pass
try:
    num_chains = int(argv[6])
except:
    num_chains = 2
try:
    num_warmup = int(argv[7])
except:
    pass
try: 
    target_accept_prob= float(argv[8])
    print(f'Target Accept Prob = {target_accept_prob}')
except Exception as ex1:
    print('EXCEPTION regarding target_accept_prob',ex1)
    target_accept_prob=0.8
try:
    cov_redshift = eval(argv[9])
except Exception as ex2:
    print('EXCEPTION regarding cov_redshift',ex2)
    cov_redshift = False
try:
    batch_bool = eval(argv[10])
except Exception as ex3:
    print('EXCEPTION regarding batch_bool',ex3)
    batch_bool = True
try:
    wa_const = eval(argv[11])
except Exception as ex4:
    print('EXCEPTION regarding wa_const',ex4)
    wa_const = False #When in doubt, allow wa to vary (along with w0, if infering wCDM cosmology)
try:
    w0_const = eval(argv[12])
except Exception as ex5:
    print('EXCEPTION regarding w0_const',ex5)
    w0_const = False  #When in doubt, allow w0 to vary (along with wa, if infering wCDM cosmology)
try:
    key_int = eval(argv[13])
except Exception as ex6:
    print('EXCEPTION regarding key_int',ex6)
    key_int = 0
try:
    GMM_zL = eval(argv[14])
except Exception as ex7:
    print('EXCEPTION regarding GMM_zL',ex7)
    GMM_zL = False
try:
    GMM_zS = eval(argv[15])
except Exception as ex8:
    print('EXCEPTION regarding GMM_zS',ex8)
    GMM_zS = False
    
print(f'Using {num_samples} samples with {num_warmup} as a warmup, and {num_chains} chains in total.')
print(f'Assuming the cosmology type is {cosmo_type}')
print(f'Full Covariance matrix setting set to {cov_redshift}')

H0_fid = 70
db_in = pd.read_csv(filein)
numpyro.enable_x64(True)

#Use the true redshifts when not allowing for photometry, 
#otherwise this biases the results as the error is already built into zL_/zS_obs:
if photometric:
    zL_to_use = jnp.array(db_in['zL_obs'])
    zS_to_use = jnp.array(db_in['zS_obs'])
else:
    zL_to_use = jnp.array(db_in['zL_true'])
    zS_to_use = jnp.array(db_in['zS_true'])

#
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
sampler_S = run_MCMC(photometric = photometric,
                    contaminated = contaminated,
                    cosmo_type = cosmo_type,
                    zL_obs = zL_to_use,
                    zS_obs = zS_to_use,
                    sigma_zL_obs = jnp.array(db_in['sigma_zL_obs']),
                    sigma_zS_obs = jnp.array(db_in['sigma_zS_obs']),
                    r_obs = jnp.array(db_in['r_obs_contam']),
                    sigma_r_obs = jnp.array(db_in['sigma_r_obs']),
                    sigma_r_obs_2 = 1000*jnp.max(jnp.array(db_in['sigma_r_obs'])),
                    P_tau = jnp.array(db_in['P_tau']),
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
                    GMM_zS=GMM_zS)

a=JAX_samples_to_dict(sampler_S,separate_keys=True,cosmo_type=cosmo_type,wa_const=wa_const,w0_const=w0_const)
db_JAX = pd.DataFrame(a)
db_JAX.to_csv(fileout,index=False)
print('File search:',file_search)
print(f'Saving warmup to {fileout_warmup}')
print(f'Saving samples to {fileout}')
print('RANDOM OUTPUT SAMPLES:',set(np.random.choice((sampler_S.get_samples(True)['Ode']).flatten(),size=20)))

