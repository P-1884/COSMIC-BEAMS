print('Loading Packages')
import os
os.environ['JAX_PLATFORMS'] = 'cpu'
import numpyro
numpyro.enable_x64(True)
from numpyro.infer import MCMC, NUTS, HMC, HMCECS
from numpyro import distributions as dist, infer
from squash_walkers import squash_walkers
from scipy.stats import truncnorm, norm
from numpyro.diagnostics import summary
from mcmcfunctions_SL import mcmc_SL
from jax import random,grad, jit
import matplotlib.pyplot as pl
from jax.random import PRNGKey
from importlib import reload
from subprocess import run
import jax.numpy as jnp
from tqdm import tqdm
import pandas as pd
import numpy as np
import importlib
import pickle
import emcee
import glob
import sys
import jax
import os
from numpyro.infer.initialization import init_to_mean #init_to_value
jax.config.update("jax_traceback_filtering", "off")
class toy_redshift_model:
    def __init__(self):
        mu_true = 1.8;sigma_true = 0.9;N_systems = 10000
        self.seed=2
        np.random.seed(self.seed);self.seed+=1
        self.true_z = jnp.array(numpyro.sample("true_z", 
                            dist.Normal(mu_true,sigma_true),
                                sample_shape=(N_systems,),rng_key=PRNGKey(self.seed)))
        np.random.seed(self.seed);self.seed+=1
        # self.sigma_z_obs = jnp.array(np.array([0.001]*(len(self.true_z)//10) + [0.3]*(9*len(self.true_z)//10)))
        # np.random.seed(self.seed);self.seed+=1
        spec_indx = np.where(self.true_z<1.5)[0]
        np.random.seed(self.seed);self.seed+=1
        spec_indx = np.random.choice(spec_indx,size=len(self.true_z)//10,replace=False)
        # print('Spec indx',spec_indx)
        phot_indx = np.array([elem for elem in np.arange(len(self.true_z)) if elem not in spec_indx])
        self.sigma_z_obs = np.nan*np.ones(len(self.true_z))
        self.sigma_z_obs[spec_indx] = 0.001
        self.sigma_z_obs[phot_indx] = 0.3
        np.random.seed(self.seed);self.seed+=1
        self.z_obs = jnp.array(numpyro.sample("z_obs", 
                            dist.Normal(loc=self.true_z,scale=self.sigma_z_obs),
                                sample_shape=(1,),rng_key=PRNGKey(self.seed))).flatten()
        assert self.sigma_z_obs.shape==self.z_obs.shape
        assert self.true_z.shape==self.z_obs.shape
        print('True mean,std',jnp.mean(self.true_z),jnp.std(self.true_z))
        print('Observed mean,std',jnp.mean(self.z_obs),jnp.std(self.z_obs))
        np.random.seed(self.seed);self.seed+=1
    def infer_parent_dist(self):
        mu = numpyro.sample('mu', dist.Uniform(0, 3))
        sigma = numpyro.sample('sigma', dist.LogUniform(0.1,2))
        zL = numpyro.sample('zL',dist.Uniform(low = self.z_obs-20,high = self.z_obs+20))
        L_0 = np.sum(dist.Normal(loc=mu, scale=sigma).log_prob(zL)+\
                     dist.Normal(loc=zL, scale=self.sigma_z_obs).log_prob(self.z_obs))
        # if z_spec is not None:
        #     L_0 += np.sum(dist.Normal(loc=mu,scale=sigma).log_prob(z_spec))
        # jax.debug.print('zl {zL}',zL=zL)
        jax.debug.print('mu {mu}',mu=mu)
        # jax.debug.print('Spec: {L_spec}',L_spec = jnp.sum(L_0[self.spec_indx]))
        # jax.debug.print('Av diff, Spec: {diff_spec}',diff_spec = jnp.mean((zL-self.true_z)[self.spec_indx]))
        # jax.debug.print('Phot: {L_phot}',L_phot = jnp.sum(L_0[self.phot_indx]))
        # jax.debug.print('Av diff, Phot: {diff_phot}',diff_phot = jnp.mean((zL-self.true_z)[self.phot_indx]))
        #L_0 = jnp.sum(L_0)
        L = numpyro.factor('Likelihood',L_0) 
    def toy_parent_redshift_MCMC(self):
        # self.spec_indx = self.sigma_z_obs==0.01
        # self.phot_indx = self.sigma_z_obs==0.4
        nuts_kernel = NUTS(self.infer_parent_dist,target_accept_prob=0.8)#,init_strategy = init_to_mean)
        mcmc = MCMC(nuts_kernel, num_samples=1000, num_warmup=1000,num_chains=3)
        self.seed+=1
        mcmc.run(random.PRNGKey(self.seed))
                                   #self.true_z[self.sigma_z_obs==np.nan])
        posterior_samples = mcmc.get_samples(True)
        return posterior_samples

argv = sys.argv
print(argv)
if argv[1] not in ['True','False']:
    argv[1]='False'

if eval(argv[1]): #Argv should be True to run the MCMC
    toy_redshift_model_0 = toy_redshift_model()
    sampler = toy_redshift_model_0.toy_parent_redshift_MCMC()

    N_saved = len(glob.glob('./test_sampler*'))
    file_out = f"./test_sampler_{N_saved}.pickle"
    print(f'Saving sampler as {file_out}')
    with open(file_out, "wb") as output_file:
        pickle.dump(sampler, output_file)