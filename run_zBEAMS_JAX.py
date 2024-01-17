#from tensorflow.python.client import device_lib
import sys
argv = sys.argv
print('ARGS',argv)

import numpyro
import jax
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
import arviz as az
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
    print('EXCEPTION',ex1)
    target_accept_prob=0.8

print(f'Using {num_samples} samples with {num_warmup} as a warmup, and {num_chains} chains in total.')
print(f'Assuming the cosmology type is {cosmo_type}')
H0_fid = 70
#numpyro.enable_x64(True)

db_in = pd.read_csv(filein)
sampler_S = run_MCMC(photometric = photometric,
                    contaminated = contaminated,
                    cosmo_type = cosmo_type,
                    zL_obs = jnp.array(db_in['zL_obs']),
                    zS_obs = jnp.array(db_in['zS_obs']),
                    sigma_zL_obs = jnp.array(db_in['sigma_zL_obs']),
                    sigma_zS_obs = jnp.array(db_in['sigma_zS_obs']),
                    r_obs = jnp.array(db_in['r_obs_contam']),
                    sigma_r_obs = jnp.array(db_in['sigma_r_obs']),
                    sigma_r_obs_2 = 1000*jnp.max(jnp.array(db_in['sigma_r_obs'])),
                    P_tau = 0.99*jnp.array(db_in['P_tau']),
                    num_warmup = num_warmup,
                    num_samples = num_samples,
                    num_chains = num_chains,
                    H0=H0_fid,
                    target_accept_prob=target_accept_prob)

def JAX_samples_to_dict(sampler,separate_keys=False,cosmo_type=''):
    key_list = sampler.get_samples().keys()
    sample_dict = {}
    for k_i in key_list:
        if 'unscaled' in k_i: continue
        if not separate_keys: 
            assert sampler.get_samples()[k_i].shape[1]==1 and len(sampler.get_samples()[k_i].shape)==2
            sample_dict[k_i] = sampler.get_samples()[k_i].T[0]
        else: 
            print(k_i,sampler.get_samples(True)[k_i].shape)
            if k_i not in ['Ok','zL','zS']: 
                if k_i=='Ok': assert sampler.get_samples(True)[k_i].shape[2]==1 and len(sampler.get_samples(True)[k_i].shape)==3
                if k_i in ['zL','zS']: assert sampler.get_samples(True)[k_i].shape[2]==1 and len(sampler.get_samples(True)[k_i].shape)==4
            for c_i in range(sampler.get_samples(True)[k_i].shape[0]):
                try:
                    if k_i not in ['zL','zS']:
                        sample_dict[f'{k_i}_{c_i}'] = sampler.get_samples(True)[k_i][c_i,:,0]
                        print('Saved shape 1:',k_i,c_i,sampler.get_samples(True)[k_i][c_i,:,0].shape)
                    #May require this if using photometric redshifts
                    if k_i in ['zL','zS']:
                        for z_i in range(sampler.get_samples(True)[k_i].shape[-1]):
                            sample_dict[f'{k_i}_{z_i}_{c_i}'] = sampler.get_samples(True)[k_i][c_i,:,z_i]
                            print('Saved shape 2:',f'{k_i}_{z_i}_{c_i}',sampler.get_samples(True)[k_i][c_i,:,z_i].shape)
                except:
                    #May require this exception if using FlatwCDM or FlatLambdaCDM
                    print('Exception here',k_i,cosmo_type)
                    assert (k_i=='Ok') or (k_i in ['w','wa'] and cosmo_type in ['LambdaCDM','FlatLambdaCDM'])
                    sample_dict[f'{k_i}_{c_i}'] = sampler.get_samples(True)[k_i][c_i,:]
                    print('Saved shape 3:',f'{k_i}_{c_i}',sampler.get_samples(True)[k_i][c_i,:].shape)
    return sample_dict

a=JAX_samples_to_dict(sampler_S,separate_keys=True,cosmo_type=cosmo_type)
db_JAX = pd.DataFrame(a)
fileout = f'./chains/SL_orig_{filein.split("/")[2]}'+\
          f'_ph_{photometric}_con_{contaminated}'+\
          f'_{cosmo_type}_JAX_chains_{time.time()}.csv'

print(f'Saving samples to {fileout}')
db_JAX.to_csv(fileout,index=False)

