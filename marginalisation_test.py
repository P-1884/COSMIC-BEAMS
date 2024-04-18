from scipy.stats import norm
import subprocess
import matplotlib.pyplot as pl
import pandas as pd
import numpy as np
import emcee

from numpyro import distributions as dist
from numpyro import sample,factor,plate,subsample
import jax.numpy as jnp
from jax.random import PRNGKey
from numpyro.infer import MCMC,NUTS,HMC,HMCECS
import time

def marginalisation_test(z_obs,r_obs,sigma_z_obs,sigma_r_obs,N_steps=5000,verbose=False):
    def likelihood_x(z_true,theta,z_obs,r_obs,sigma_z_obs,sigma_r_obs,mu_z_pop,sigma_z_pop):
        if sigma_z_obs<=0: return -np.inf
        if sigma_r_obs<=0: return -np.inf
        if sigma_z_pop<=0: return -np.inf
        p1 = np.sum(norm(z_true,sigma_z_obs).logpdf(z_obs)) #Use (natural) logpdf to reduce rounding errors
        p2 = np.sum(norm(r_obs,sigma_r_obs).logpdf(r_func(z_true,theta)))
        p3 = np.sum(norm(z_true,sigma_z_pop).logpdf(mu_z_pop))
        l_i =  p1+p2+p3 #Sum rather than product as taking logarithm
        return l_i
    def prior_xy(theta,mu_z_pop,sigma_z_pop,z_true,z_min,z_max):
        if theta<=0 or theta>1: return -np.inf
        if mu_z_pop<=0 or mu_z_pop>100: return -np.inf
        if sigma_z_pop<=0 or sigma_z_pop>20: return -np.inf
        if (z_true<z_min).any() or (z_true>z_max).any(): return -np.inf
        else: return 0
    def posterior_x(t_mu_sigma_z,z_obs,r_obs,sigma_z_obs,sigma_r_obs,N_obs,z_min,z_max):
        theta=t_mu_sigma_z[0];mu_z_pop=t_mu_sigma_z[1]
        sigma_z_pop = t_mu_sigma_z[2];z_true = t_mu_sigma_z[3:3+N_obs]
        prior_i = prior_xy(theta,mu_z_pop,sigma_z_pop,z_true,z_min,z_max)
        likelihood_i = likelihood_x(z_true,theta,z_obs,r_obs,sigma_z_obs,sigma_r_obs,mu_z_pop,sigma_z_pop)
        lp = prior_i+likelihood_i
        if verbose: print(lp,prior_i,likelihood_i)
        return lp
    N_obs = len(z_obs)
    N_walkers = 2*(N_obs + 3)
    z_min = np.min(z_obs)-4*np.max(sigma_z_obs)
    z_max = np.max(z_obs)+4*np.max(sigma_z_obs)
    cur_theta = np.random.uniform(0,1,size=(N_walkers,1))
    cur_mu = np.random.uniform(0,100,size=(N_walkers,1))
    cur_sigma = np.random.uniform(0,20,size=(N_walkers,1))
    cur_z_true = np.random.uniform(z_min,z_max,size=(N_walkers,N_obs))
    cur_xy = np.concatenate([cur_theta,cur_mu,cur_sigma,cur_z_true],axis=1)
    print(f'Running MCMC with {N_obs} observations, {N_walkers} chains and {N_steps} steps')
    print(f'z integration from {z_min} to {z_max}')
    sampler_x2 = emcee.EnsembleSampler(N_walkers, 3+N_obs, posterior_x,args = [z_obs,r_obs,sigma_z_obs,sigma_r_obs,N_obs,
                                                                               z_min,z_max])
    sampler_x2.run_mcmc(cur_xy, N_steps, progress=True)
    return sampler_x2,N_walkers

def r_func(z,theta):
    return theta*(z**2)

def posterior_x(t_mu_sigma_z,z_obs,r_obs,sigma_z_obs,sigma_r_obs,N_obs,z_min,z_max):
        def likelihood_x(z_true,theta,z_obs,r_obs,sigma_z_obs,sigma_r_obs):
            if sigma_z_obs<=0: return -np.inf
            if sigma_r_obs<=0: return -np.inf
            p1 = np.sum(norm(z_true,sigma_z_obs).logpdf(z_obs)) #Use (natural) logpdf to reduce rounding errors
            p2 = np.sum(norm(r_obs,sigma_r_obs).logpdf(r_func(z_true,theta)))
            l_i =  p1+p2 #Sum rather than product as taking logarithm
            return l_i
        def prior_xy(theta,z_true,z_min,z_max):
            if theta<=0 or theta>1: return -np.inf
            if (z_true<z_min).any() or (z_true>z_max).any(): return -np.inf
            else: return 0
        theta = t_mu_sigma_z[0]
        z_true = t_mu_sigma_z[1:1+N_obs]
        prior_i = prior_xy(theta,z_true,z_min,z_max)
        likelihood_i = likelihood_x(z_true,theta,z_obs,r_obs,sigma_z_obs,sigma_r_obs)
        lp = prior_i+likelihood_i
        #if verbose: print(lp,prior_i,likelihood_i)
        return lp
    
def marginalisation_test_no_parent(z_obs,r_obs,sigma_z_obs,sigma_r_obs,N_steps=5000,verbose=False,sampler_kwargs={}):
    #As above, but instead without the hierarchical parameters mu_z and sigma_z, i.e. assuming there is only
    #one z_true value and many independent observations of it, z_obs.
    N_obs = len(z_obs)
    N_walkers = 2*(N_obs + 1)
    cur_theta = np.random.uniform(0.0,1.0,size=(N_walkers,1))
    z_min = np.min(z_obs)-4*np.max(sigma_z_obs)
    z_max = np.max(z_obs)+4*np.max(sigma_z_obs)
    cur_z_true = np.random.uniform(z_min,z_max,size=(N_walkers,N_obs))
    cur_xy = np.concatenate([cur_theta,cur_z_true],axis=1)
    print(f'Running MCMC with {N_obs} observations, {N_walkers} chains and {N_steps} steps')
    print(f'z integration from {z_min} to {z_max}')
    sampler_x2 = emcee.EnsembleSampler(N_walkers, 1+N_obs, posterior_x, args = [z_obs,r_obs,sigma_z_obs,sigma_r_obs,
                                                                                N_obs,z_min,z_max],
                                       **sampler_kwargs)
    sampler_x2.run_mcmc(cur_xy, N_steps, progress=True)
    return sampler_x2,N_walkers

def j_marginalisation(z_obs,r_obs,sigma_z_obs,sigma_r_obs,key=None,quick_return=False,batch=False):
        #What is the difference between sampling z_true uniformly then adding a dist.Normal term in the likelihood
        #versus sampling z_true Normally, and not including it in the likelihood? 
        #
        #
        theta = sample("theta", dist.Uniform(0,1),sample_shape=(1,),rng_key=key)
        mu_z = jnp.squeeze(sample("mu_z", dist.Uniform(0,40),sample_shape=(1,),rng_key=key))
        sigma_z = jnp.squeeze(sample("sigma_z", dist.LogUniform(1,20),sample_shape=(1,),rng_key=key))
        z_min = z_obs-3*sigma_z_obs
        z_max = z_obs+3*sigma_z_obs
        z_true = sample('z_true',dist.Uniform(z_min,z_max),sample_shape=(1,),rng_key=key).flatten()
        P1_a = dist.Normal(z_true,sigma_z_obs).log_prob(z_obs)
        P1 = np.sum(P1_a)
        P2 = np.sum(dist.Normal(r_obs,sigma_r_obs).log_prob(r_func(z_true,theta)))
        P3 = np.sum(dist.Normal(z_true,sigma_z).log_prob(mu_z))
        if quick_return:
             print('SHAPES',z_min.shape,z_max.shape,z_true.shape,P1_a.shape)
             return
        if not batch:
            L = factor('Likelihood',P1 + P2 + P3)
        if batch:
            with plate("N", z_obs.shape[0], subsample_size=z_obs.shape[0]//2):
                batch_P1 = subsample(P1,event_dim=0)
                batch_P2 =  subsample(P2,event_dim=0)
                batch_P3  = subsample(P3,event_dim=0)
                L = factor("Likelihood",batch_P1+batch_P2+batch_P3)

def run_MCMC_marginalisation(z_obs,r_obs,sigma_z_obs,sigma_r_obs,
            num_warmup = 200,num_samples=1000,num_chains=2,
            target_accept_prob=0.8,warmup_file=np.nan,batch = False):
    key = PRNGKey(0)
    print(f'Target Accept Prob: {target_accept_prob}')
    model_args = {'z_obs':z_obs,'r_obs':r_obs,'sigma_z_obs':sigma_z_obs,'sigma_r_obs':sigma_r_obs,'batch':batch}
    st = time.time()
    j_marginalisation(**model_args,key=key,quick_return=True)
    mt=time.time()
    j_marginalisation(**model_args,key=key,quick_return=True)
    et=time.time()
    print('Uncompiled time',mt-st)
    print('Compiled time',et-mt)
    if batch:
        inner_kernel = NUTS(model = j_marginalisation,target_accept_prob = target_accept_prob)         
        kernel = HMCECS(inner_kernel, num_blocks=100)
    else:
        kernel = NUTS(model = j_marginalisation,target_accept_prob = target_accept_prob)
    sampler_0 = MCMC(kernel,
                    num_warmup=num_warmup,
                    num_samples=num_samples,
                    num_chains=num_chains,
                    progress_bar=True)
    sampler_0.warmup(key,**model_args,collect_warmup=True)
    ##
    # warmup_dict = JAX_samples_to_dict(sampler_0,separate_keys=True,cosmo_type=cosmo_type)
    # db_JAX_warmup = pd.DataFrame(warmup_dict)
    # db_JAX_warmup.to_csv(warmup_file,index=False)
    ##
    sampler_0.run(key,**model_args,key=key)
    return sampler_0