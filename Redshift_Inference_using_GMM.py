print('Loading Packages')
# from zbeamsfunctions_SL import likelihood_SL,likelihood_spec_contam_SL,likelihood_phot_contam_SL,likelihood_phot_SL,r_SL
from astropy.cosmology import LambdaCDM,FlatLambdaCDM,wCDM,FlatwCDM,w0waCDM
from zbeamsfunctions import mu_w,likelihood,likelihood_spec
# from mcmcfunctions_SL_JAX import j_likelihood_SL,run_MCMC
from Lenstronomy_Cosmology import Background, LensCosmo
from scipy.stats import multivariate_normal as MVN
from mcmcfunctions import mcmc,mcmc_spec,mcmc_phot
from numpyro import distributions as dist, infer
from numpyro.infer import MCMC, NUTS, HMC, HMCECS
from squash_walkers import squash_walkers
from scipy.stats import truncnorm, norm
from numpyro.diagnostics import summary
import matplotlib.patches as mpatches
# from mcmcfunctions_SL import mcmc_SL
import matplotlib.lines as mlines
# from cosmology_JAX import j_r_SL
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
from numpyro.infer.initialization import init_to_mean #init_to_value

try:importlib.reload(sys.modules['mcmcfunctions_SL'])
except Exception as ex: print(f'Cannot reload: {ex}')

'''
Have shown the JAX and emcee modules give answers in agreement for a very simple cosmology. Need to further
demonstrate this with more complex cosmologies (inc w0wa cosmology which emcee doesn't yet have?), but most
importantly including contamination + photometry.
'''

#!python3 -m pip install funsor
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

def gaussian_inference(zL_obs,zL_sigma,likelihood_check=False,likelihood_input = {}):
    # Define the priors for the mean and standard deviation
    if likelihood_check:
        mu1,mu2 = likelihood_input['mu'],likelihood_input['mu2']
        sigma1,sigma2 = likelihood_input['sigma'],likelihood_input['sigma2']
        w = likelihood_input['w']
        zL = likelihood_input['zL']
    else:
        mu1 = numpyro.sample('mu', dist.Uniform(0, 4))
        sigma1 = numpyro.sample('sigma', dist.LogUniform(0.01,5))
        mu2 = numpyro.sample('mu2', dist.Uniform(0,4))
        sigma2 = numpyro.sample('sigma2', dist.LogUniform(0.01,5))
        w = numpyro.sample('w', dist.Uniform(0,1)) # Not - First component is the largest - seemed to not work when imposing this.
        zL_obs_low_lim = jnp.array(zL_obs-10*zL_sigma) #0*jnp.ones(len(zL_obs)) #
        # zL_obs_low_lim = zL_obs_low_lim*(zL_obs_low_lim>0) #Minimum value is 0
        zL_obs_up_lim = jnp.array(zL_obs+10*zL_sigma) #20*jnp.ones(len(zL_obs)) #
        zL = numpyro.sample('zL',dist.Uniform(low = zL_obs_low_lim,high = zL_obs_up_lim))#,sample_shape=(1,)).flatten()
    # jax.debug.print('w {w}',w=w)
    # jax.debug.print('zL {zL}',zL=zL)
    # jax.debug.print('Upper Lim {zL_obs_up_lim}',zL_obs_up_lim=zL_obs_up_lim)
    # Define the likelihood of the observed data
    L_0 = dist.Mixture(dist.Categorical(probs=jnp.array([w,1-w])),
                    #    [dist.TruncatedNormal(mu1, sigma1,low=0),dist.TruncatedNormal(mu2, sigma2,low=0)
                        [dist.Normal(mu1, sigma1),dist.Normal(mu2, sigma2)
                            ]).log_prob(zL)+\
                    dist.TruncatedNormal(zL, zL_sigma, low=0).log_prob(zL_obs)
    if likelihood_check: return L_0
    L = numpyro.factor('Likelihood',L_0)
                    # dist.Normal(zL, zL_sigma).log_prob(zL_obs))

def GMM_mcmc(zL_obs,zL_sigma):
    # Run the inference algorithm
    nuts_kernel = NUTS(gaussian_inference,init_strategy = init_to_mean)
    mcmc = MCMC(nuts_kernel, num_samples=2000, num_warmup=2000,num_chains=2)
    mcmc.run(random.PRNGKey(1),zL_obs,zL_sigma)

    # Get the posterior samples
    posterior_samples = mcmc.get_samples(True)

    return posterior_samples

# try:
redshift_type = sys.argv[1]
db_filename = sys.argv[2]

db_out = pd.read_csv(f'/mnt/users/hollowayp/zBEAMS/databases/{db_filename}') #real_paltas_population_TP_10000_FP_0_Spec_0_P_1.0.csv
b = GMM_mcmc(jnp.array(db_out[f'{redshift_type}_obs']),jnp.array(db_out[f'sigma_{redshift_type}_obs']))
for k_i in b.keys():
    print(k_i,np.mean(b[k_i]),np.std(b[k_i]),b[k_i].shape)

GMM_dict = {}
for k_i in ['mu','mu2','sigma','sigma2','w','zL']:
    print(b[k_i].shape)
    if k_i=='zL':
        for z_ii in range(100): #Only save 100 redshifts
            for c_i in range(b[k_i].shape[0]):
                GMM_dict[f'{k_i}_{c_i}_{z_ii}'] = b[k_i][c_i,:,z_ii]
                print('saving z',GMM_dict[f'{k_i}_{c_i}_{z_ii}'].shape)
    else:
        for c_i in range(b[k_i].shape[0]):
            GMM_dict[f'{k_i}_{c_i}'] = b[k_i][c_i,:]
            print(GMM_dict[f'{k_i}_{c_i}'].shape)

file_out = f'/mnt/zfsusers/hollowayp/zBEAMS/GMM_redshifts_{redshift_type}_{db_filename.replace(".csv","")}_fit'
N_previous_files = len(glob.glob(file_out+'*'))
file_out = f'{file_out}_{N_previous_files}.csv'
print(f'Saving to {file_out}')
pd.DataFrame(GMM_dict).to_csv(file_out)
# except Exception as ex:
#     print('Exception',ex)