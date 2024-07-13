print('Loading Packages')
#from zbeamsfunctions_SL import likelihood_SL,likelihood_spec_contam_SL,likelihood_phot_contam_SL,likelihood_phot_SL,r_SL
from astropy.cosmology import LambdaCDM,FlatLambdaCDM,wCDM,FlatwCDM,w0waCDM
from zbeamsfunctions import mu_w,likelihood,likelihood_spec
#from mcmcfunctions_SL_JAX import j_likelihood_SL,run_MCMC
from Lenstronomy_Cosmology import Background, LensCosmo
from scipy.stats import multivariate_normal as MVN
from mcmcfunctions import mcmc,mcmc_spec,mcmc_phot
from numpyro import distributions as dist, infer
from numpyro.infer import MCMC, NUTS, HMC, HMCECS
from squash_walkers import squash_walkers
from scipy.stats import truncnorm, norm
from numpyro.diagnostics import summary
import matplotlib.patches as mpatches
from mcmcfunctions_SL import mcmc_SL
import matplotlib.lines as mlines
from cosmology_JAX import j_r_SL
from jax import random,grad, jit
import matplotlib.pyplot as pl
from subprocess import run
import jax.numpy as jnp
import jax_cosmo as jc
from tqdm import tqdm
import scipy.sparse
import pandas as pd
import arviz as az
import numpy as np
import importlib
import numpyro
import corner
import emcee
import glob
import sys
import jax
#np.random.seed(1)

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
def Gaussian_truncated_at_zero_0(loc,scale,size,lower_lim = None):
    #Do NOT put a constant random seed here, or things will end up correlated where they shouldn't be!!
    if lower_lim is None: return truncnorm(a=-loc/scale,b=np.inf,loc=loc,scale=scale).rvs(size=size)
    else: return truncnorm(a=-(loc-lower_lim)/scale,b=np.inf,loc=loc,scale=scale).rvs(size=size)

from numpyro import distributions as dist,infer
from numpyro.infer import MCMC,NUTS,HMC,HMCECS
from numpyro import sample, subsample
from jax.random import PRNGKey
import time

def j_r_trunc_test(x,A,linear=True, Inc_source = False,y = jnp.array([])):
    if linear:
        if Inc_source: return A*(x**2+y**2)   
        else: return A*x**2
    else: 
        if Inc_source: return (A**0.5)*(x**2+y**2) 
        else: return (A**0.5)*x**2

def j_likelihood_SL_truncation_test(x_obs,r_obs,sigma_r_obs,key=None,
                                    quick_return=False,lower_truncation = None,linear = True,
                                    photometric = False,sigma_x_obs = None,
                                    Inc_source = False, y_obs = jnp.array([]), sigma_y_obs = jnp.array([]),
                                    return_likelihood=False,input_dict = {},
                                    truncate_y_at_x = True):
    print('Truncate y at x:',truncate_y_at_x)
    if not return_likelihood:
        if photometric:
            x = numpyro.sample('X',dist.Uniform(low = 0,high = 4*jnp.ones(len(x_obs))),sample_shape=(1,),rng_key=key).flatten()
            mu_x_pop = jnp.squeeze(numpyro.sample("mu_x_pop", dist.Uniform(0,1),sample_shape=(1,),rng_key=key))
            sigma_x_pop = jnp.squeeze(numpyro.sample("sigma_x_pop", dist.Uniform(0.1,1),sample_shape=(1,),rng_key=key))
            if Inc_source:
                if truncate_y_at_x: 
                    y = numpyro.sample('Y',dist.Uniform(low = x,high = 6*jnp.ones(len(x_obs))),sample_shape=(1,),rng_key=key).flatten()
                else:
                    y = numpyro.sample('Y',dist.Uniform(low = 0,high = 6*jnp.ones(len(x_obs))),sample_shape=(1,),rng_key=key).flatten()
                mu_y_pop = jnp.squeeze(numpyro.sample("mu_y_pop", dist.Uniform(0,1),sample_shape=(1,),rng_key=key))
                sigma_y_pop = jnp.squeeze(numpyro.sample("sigma_y_pop", dist.Uniform(0.1,1),sample_shape=(1,),rng_key=key))
                #assert (y>x).all()
            else: y = jnp.array([])
        else:
            x = x_obs #Just like for spectroscopic redshifts.
            if Inc_source: y = y_obs
        OM = jnp.squeeze(sample("OM", dist.Uniform(0,1),sample_shape=(1,),rng_key=key))
    else:
        x = input_dict['x'];mu_x_pop = input_dict['mu_x_pop'];sigma_x_pop = input_dict['sigma_x_pop']
        y = input_dict['y'];mu_y_pop = input_dict['mu_y_pop'];sigma_y_pop = input_dict['sigma_y_pop']
        OM = input_dict['OM']
    r_theory = j_r_trunc_test(x,OM,linear=linear, Inc_source = Inc_source, y = y)
    if quick_return: return
    if photometric:      
        prob_1 = dist.TruncatedNormal(r_theory, sigma_r_obs, low = lower_truncation).log_prob(r_obs)
        #if lower_truncation==0:
        print('Applying truncation')
        prob_1 = jnp.where(r_obs<0,-np.inf,prob_1)
        prob_2 = dist.TruncatedNormal(x, sigma_x_obs, low = 0).log_prob(x_obs)
        prob_2 = jnp.where(x_obs<0,-np.inf,prob_2)
        prob_3 = dist.TruncatedNormal(mu_x_pop, sigma_x_pop,low = 0).log_prob(x)
        prob_3 = jnp.where(x<0,-np.inf,prob_3)
        if Inc_source:
            # assert (y_obs>=x_obs).all()
            prob_4 = dist.TruncatedNormal(y, sigma_y_obs,low = 0).log_prob(y_obs)#, low = x_obs).log_prob(y_obs)
            prob_4 =  jnp.where(y_obs<0,-np.inf,prob_4)
            if truncate_y_at_x:
                prob_5 = dist.TruncatedNormal(mu_y_pop, sigma_y_pop,low = x).log_prob(y)
                prob_5 = jnp.where(y<x,-np.inf,prob_5)
            else:
                prob_5 = dist.TruncatedNormal(mu_y_pop, sigma_y_pop,low = 0).log_prob(y)
                prob_5 = jnp.where(y<0,-np.inf,prob_5)
            if False:
                try:
                    print('P1',prob_1,np.isinf(prob_1).any())
                    print('P2',prob_2,np.isinf(prob_2).any())
                    print('P3',prob_3,np.isinf(prob_3).any())
                    print('P4',prob_4,np.isinf(prob_4).any())
                    for inf_indx in [np.where(np.isinf(prob_4)),np.where(np.isinf(prob_5))]:
                        print('Outputs:')
                        print('y',y[inf_indx][0:5])
                        print('y_obs',y_obs[inf_indx][0:5])
                        print('x',x[inf_indx][0:5])
                        print('x_obs',x_obs[inf_indx][0:5])
                        print('sigma_y_obs',sigma_y_obs)
                        print('P5',prob_5,np.isinf(prob_5).any())
                except: pass
            if return_likelihood: 
                print(prob_1,prob_2,prob_3,prob_4,prob_5)
                return prob_1 + prob_2 + prob_3 + prob_4 + prob_5
            L = numpyro.factor("Likelihood",prob_1 + prob_2 + prob_3 + prob_4 + prob_5)
        else:
            L = numpyro.factor("Likelihood",prob_1 + prob_2 + prob_3)
    else: sample("Likelihood", dist.TruncatedNormal(r_theory, sigma_r_obs, low=lower_truncation), obs=r_obs)

def run_MCMC_truncation_test(x_obs,r_obs,sigma_r_obs,
            num_warmup = 2000,num_samples=2000,num_chains=2,
            target_accept_prob=0.8,warmup_file=np.nan,lower_truncation=None,linear = True, photometric = False,
            sigma_x_obs = None, Inc_source = False, y_obs = jnp.array([]), sigma_y_obs = jnp.array([]),
            truncate_y_at_x = True):
    model_args = {'x_obs':x_obs,
                  'r_obs':r_obs,'sigma_r_obs':sigma_r_obs,'lower_truncation':lower_truncation,'linear':linear,
                  'photometric':photometric,'sigma_x_obs':sigma_x_obs, 'Inc_source':Inc_source,'y_obs':y_obs,'sigma_y_obs':sigma_y_obs,
                  'truncate_y_at_x':truncate_y_at_x}
    key = PRNGKey(0)
    j_likelihood_SL_truncation_test(**model_args,key=key,quick_return=True)
    j_likelihood_SL_truncation_test(**model_args,key=key,quick_return=True)
    outer_kernel =  NUTS(model = j_likelihood_SL_truncation_test,target_accept_prob = target_accept_prob)
    sampler_0 = MCMC(outer_kernel,
                    num_warmup=num_warmup,
                    num_samples=num_samples,
                    num_chains=num_chains,
                    progress_bar=True)
    sampler_0.warmup(key,**model_args,collect_warmup=True,key=None)
    sampler_0.run(key,**model_args,key=None)
    return sampler_0

import sys
argv = sys.argv
N_systems = int(argv[1])
lower_truncation = eval(argv[2])
linear = eval(argv[3])
photometric = eval(argv[4])
Inc_source = eval(argv[5])
truncate_y_at_x = eval(argv[6])
print(argv,N_systems,lower_truncation,linear,photometric,Inc_source,truncate_y_at_x)
A_true = 0.3
mu_zL_pop = 0.2
mu_zS_pop = 0.6
sigma_pop = 0.5
x_trunc = truncnorm(loc=mu_zL_pop,scale=sigma_pop,a = -mu_zL_pop/sigma_pop,b = np.inf).rvs(size=N_systems)
if truncate_y_at_x:
    y_trunc = truncnorm(loc=mu_zS_pop,scale=sigma_pop,a = -(mu_zS_pop-x_trunc)/sigma_pop,b = np.inf).rvs(size=N_systems)
else:
    y_trunc = truncnorm(loc=mu_zS_pop,scale=sigma_pop,a = -mu_zS_pop/sigma_pop,b = np.inf).rvs(size=N_systems)

assert len(x_trunc)==len(y_trunc)
assert (y_trunc>=0).all()
assert (x_trunc>=0).all()
r_true_trunc = j_r_trunc_test(x_trunc,A_true,linear=linear,Inc_source = Inc_source,y = y_trunc)

if photometric: sigma_list = [0.05]
else: sigma_list = [0.05,0.1,0.2,0.3]
x_sigma_list = [0.3,0.2,0.1,0.05]
db_out = pd.DataFrame()
db_out['x_true'] = x_trunc
db_out['r_true'] = r_true_trunc
if Inc_source: db_out['y_true'] = y_trunc

for ii,sigma_i in enumerate(sigma_list):
    sigma_r_obs_trunc = sigma_i
    for jj, x_sigma_i in enumerate(x_sigma_list):
        y_sigma_i = x_sigma_i
        if photometric:
            x_obs = Gaussian_truncated_at_zero_0(x_trunc,x_sigma_i,len(x_trunc))
            if Inc_source: y_obs = Gaussian_truncated_at_zero_0(y_trunc,y_sigma_i,len(y_trunc))#,lower_lim = x_obs)
            else: y_obs = jnp.array([])
        else: x_obs = x_trunc; y_obs = y_trunc
        #if Inc_source: assert (y_obs>=x_obs).all()
        r_obs_trunc = jnp.array(Gaussian_truncated_at_zero_0(r_true_trunc,sigma_r_obs_trunc,len(r_true_trunc)))
        s_trunc = run_MCMC_truncation_test(x_obs,r_obs_trunc,sigma_r_obs_trunc,
                                        lower_truncation=lower_truncation,
                                        linear=linear,
                                        photometric = photometric,
                                        sigma_x_obs = x_sigma_i,
                                        Inc_source = Inc_source, y_obs = y_obs, sigma_y_obs = y_sigma_i,
                                        truncate_y_at_x=truncate_y_at_x)
        for chain_i in range(s_trunc.get_samples(True)['OM'].shape[0]):
            chain_dict_trunc = {}
            chain_dict_trunc[f'{sigma_i}_{chain_i}_{x_sigma_i}'] = s_trunc.get_samples(True)['OM'][chain_i,:,0]
            print(f'Generating pandas DF, chain {chain_i}')
            db_trunc = pd.DataFrame(chain_dict_trunc)
            file_out = f'/mnt/zfsusers/hollowayp/zBEAMS/databases/trunc_databases/Gauss_Toy_model_{N_systems}_{str(lower_truncation)}_{linear}_{photometric}_{Inc_source}_C{chain_i}_'+\
                f'S{sigma_i}_Sx{x_sigma_i}_'+\
                ('Tr_y_'+str(truncate_y_at_x))*(truncate_y_at_x==False)+'.csv'
            print(f'Saving to {file_out}')
            db_trunc.to_csv(file_out)
        db_out[f'x_obs_{x_sigma_i}'] = x_obs 
        db_out[f'r_obs_{sigma_r_obs_trunc}'] = r_obs_trunc
        if Inc_source: db_out[f'y_obs_{x_sigma_i}'] = y_obs
        db_out_file = f'/mnt/zfsusers/hollowayp/zBEAMS/databases/trunc_databases/Gauss_Toy_model_data_{N_systems}_{str(lower_truncation)}_{linear}_{photometric}_{Inc_source}_'+\
                f'S{sigma_i}_Sx{x_sigma_i}_'+\
                ('Tr_y_'+str(truncate_y_at_x))*(truncate_y_at_x==False)+'.csv'
        db_out.to_csv(db_out_file)