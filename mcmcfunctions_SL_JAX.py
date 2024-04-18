from astropy.cosmology import LambdaCDM,FlatLambdaCDM,wCDM,FlatwCDM,w0waCDM
from JAX_samples_to_dict import JAX_samples_to_dict
from numpyro import distributions as dist,infer
from numpyro.infer import MCMC,NUTS,HMC,HMCECS
from jax import random,grad, jit
import matplotlib.pyplot as pl
import jax.numpy as jnp
import jax_cosmo as jc
from tqdm import tqdm
import scipy.sparse
import pandas as pd
import arviz as az
import numpy as np
import numpyro
import corner
import emcee
import sys
import jax
import time
#jax.config.update("jax_enable_x64", True)

@jit
def MVN_samp(loc_0,loc_1,sig_0,sig_1,x0,x1,sigma_01,sigma_10):
        return dist.MultivariateNormal(loc=jnp.array([loc_0,loc_1]),
                        covariance_matrix=jnp.array([[sig_0**2,sigma_01**2],
                                                     [sigma_10**2,sig_1**2]])).log_prob(jnp.array([x0,x1]).T) 

@jit
def likelihood_PC(P_tau,prob_1a,prob_1b,prob_2,prob_3,prob_4a,prob_4b):
        return (jnp.log(P_tau*jnp.exp(prob_1a)*jnp.exp(prob_4a)+(1-P_tau)*jnp.exp(prob_1b)*jnp.exp(prob_4b))+prob_2+prob_3)

def j_likelihood_SL(zL_obs,zS_obs,sigma_zL_obs,sigma_zS_obs,r_obs,sigma_r_obs,sigma_r_obs_2=[np.nan],P_tau = [],cosmo_type='',
                    photometric=False,contaminated=False,H0=np.nan,key=None,
                    likelihood_check=False,likelihood_dict = {},cov_redshift=False,early_return=False):
    s8 = 0.8;n_s = 0.96;Ob=0; #Putting all the matter in dark-matter (doesn't make a difference)
    subsample_size = 7000
    if not likelihood_check:
        if photometric:
            print('Assuming photometric redshifts')
            zL_sigma = sigma_zL_obs;zS_sigma = sigma_zS_obs
            # Had introduced this (unscaled version first) as it was supposed to make MCMC faster, but not sure it did:
            # zL_unscaled = numpyro.sample('zL_unscaled',dist.TruncatedNormal(0,scale=jnp.ones(zL_obs.shape),low=0),sample_shape=(1,),
            #                         rng_key=key).flatten()
            # zS_unscaled = numpyro.sample('zS_unscaled',dist.TruncatedNormal(0,scale=jnp.ones(zS_obs.shape),low=0),sample_shape=(1,),
            #                             rng_key=key).flatten()
            # zL = numpyro.deterministic('zL',jnp.array(zL_obs)+zL_unscaled*jnp.array(zL_sigma))
            # zS = numpyro.deterministic('zS',jnp.array(zS_obs)+zS_unscaled*jnp.array(zS_sigma))
            # Removing assertion that zS has to be > zL - still bugs:
            zL = numpyro.sample('zL',dist.TruncatedNormal(jnp.array(zL_obs),zL_sigma,low=0),sample_shape=(1,),rng_key=key).flatten()
            zS = numpyro.sample('zS',dist.TruncatedNormal(jnp.array(zS_obs),zS_sigma,low=0),sample_shape=(1,),rng_key=key).flatten()
            mu_zL_g_L = jnp.squeeze(numpyro.sample("mu_zL_g_L", dist.Uniform(0,0.5),sample_shape=(1,),rng_key=key))
            mu_zS_g_L = jnp.squeeze(numpyro.sample("mu_zS_g_L", dist.Uniform(0.5,1),sample_shape=(1,),rng_key=key))
            sigma_zL_g_L = jnp.squeeze(numpyro.sample("sigma_zL_g_L", dist.Uniform(0.01,1),sample_shape=(1,),rng_key=key))
            sigma_zS_g_L = jnp.squeeze(numpyro.sample("sigma_zS_g_L", dist.Uniform(0.01,1),sample_shape=(1,),rng_key=key))
            if cov_redshift:
                sigma_01_g_L =  jnp.squeeze(numpyro.sample("sigma_01_g_L", dist.Uniform(0.01,1),sample_shape=(1,),rng_key=key))
                sigma_10_g_L =  jnp.squeeze(numpyro.sample("sigma_10_g_L", dist.Uniform(0.01,1),sample_shape=(1,),rng_key=key))
            else:
                sigma_01_g_L =  0.0;sigma_10_g_L =  0.0
        else:
            print('Assuming spectroscopic redshifts')
            zL = zL_obs #Think still need to have an error-budget when using spectroscopic redshifts?
            zS = zS_obs
        OM = jnp.squeeze(numpyro.sample("OM", dist.Uniform(0,1),sample_shape=(1,),rng_key=key))
        if cosmo_type in ['FlatLambdaCDM','FlatwCDM']:
            #Don't care about Ode, as it isn't an argument for the cosmology (OM and Ok are instead)
            print('Assuming a flat universe')
            Ok = numpyro.deterministic('Ok',0.0)
        else:
            print('Assuming the universe may have curvature')
            print('NOTE: Need to think about my priors more here - if there is no information, the Om and Ode priors are uniform, so my\n '+\
                'Ok prior would be triangular (1-(Ode+Om)), centered on 0. **This is very problematic**. Perhaps need to sample from a uniform\n '+\
                '3D distribution with some planar cuts to make Ode+Ok+Om=1?')
            Ode = jnp.squeeze(numpyro.sample("Ode", dist.Uniform(0,1),sample_shape=(1,),rng_key=key))
            Ok = numpyro.deterministic('Ok',1-(OM+Ode))
        if cosmo_type in ['LambdaCDM','FlatLambdaCDM']:
            print('Assuming universe has a cosmological constant')
            w = numpyro.deterministic('w',-1.0)
            wa = numpyro.deterministic('wa',0.0)
        else:
            print('Assuming non-trivial dark energy equation of state')
            w = jnp.squeeze(numpyro.sample("w", dist.Uniform(-6,4),sample_shape=(1,),rng_key=key)) #Physicality constraints
            wa = jnp.squeeze(numpyro.sample("wa", dist.Uniform(-3,1),sample_shape=(1,),rng_key=key)) #Matching Tian's constraints for now
    if likelihood_check:
            OM = likelihood_dict['OM'];Ok = likelihood_dict['Ok']
            w = likelihood_dict['w'];wa=likelihood_dict['wa'];
            r_obs =likelihood_dict['r_obs'];sigma_r_obs = likelihood_dict['sigma_r_obs'];
            zL = likelihood_dict['zL'];zS = likelihood_dict['zS']
            zL_obs = likelihood_dict['zL_obs'];zS_obs = likelihood_dict['zS_obs']
            zL_sigma = likelihood_dict['zL_sigma'];zS_sigma = likelihood_dict['zS_sigma']
            mu_zL_g_L,mu_zS_g_L = likelihood_dict['mu_zL_g_L'],likelihood_dict['mu_zS_g_L']
            sigma_zL_g_L,sigma_zS_g_L = likelihood_dict['sigma_zL_g_L'],likelihood_dict['sigma_zS_g_L']
            sigma_01_g_L,sigma_10_g_L = likelihood_dict['sigma_01_g_L'],likelihood_dict['sigma_10_g_L']
    cosmo = jc.Cosmology(Omega_c=OM, h=H0/100, Omega_k=Ok, w0=w,
                         Omega_b=Ob, wa=wa, sigma8=s8, n_s=n_s)
    if cosmo_type in ['FlatLambdaCDM','FlatwCDM']: r_theory = j_r_SL_flat(zL,zS,cosmo)
    else: r_theory = j_r_SL(zL,zS,cosmo)
    if early_return: return
    if contaminated and not photometric:
        P_tau = P_tau.astype('float') #Needs to be a float for dist.Categorical to work
        print('Assuming contaminated, with spectroscopic redshifts')
        assert not np.isnan(sigma_r_obs_2)
        print('NOTE: Need to come up with a test function (e.g. known likelihood) to see what this is actually doing, rather than just\n'+\
            ' assuming that because it gives the right answer it must be correct.')
#         prob_1 = dist.Mixture(dist.Categorical(jnp.array([P_tau, 1-P_tau]).T),
#                             [dist.Normal(r_theory, sigma_r_obs),
#                              dist.Normal(r_theory, sigma_r_obs_2)]).log_prob(r_obs)
#         L = numpyro.factor("Likelihood",prob_1)
        with numpyro.plate("N", r_theory.shape[0], subsample_size=subsample_size):
            batch_r_obs = numpyro.subsample(r_obs,event_dim=0)
            batch_P_tau =  numpyro.subsample(P_tau,event_dim=0)
            batch_r_theory = numpyro.subsample(r_theory,event_dim=0)
            batch_sigma_r_obs = numpyro.subsample(sigma_r_obs,event_dim=0)
            batch_sigma_r_obs_2 = numpyro.subsample(sigma_r_obs_2,event_dim=0)
            batch_prob_1 = dist.Mixture(dist.Categorical(jnp.array([batch_P_tau, 1-batch_P_tau]).T),
                            [dist.Normal(batch_r_theory, batch_sigma_r_obs),
                             dist.Normal(batch_r_theory, batch_sigma_r_obs_2)]).log_prob(batch_r_obs)
            L = numpyro.factor("Likelihood",batch_prob_1)
    elif photometric and not contaminated:
        print('Assuming not contaminated, with photometric redshifts')
        print('NOTE: Need to come up with a test function (e.g. known likelihood) to see what this is actually doing, rather than just\n'+\
            ' assuming that because it gives the right answer it must be correct.')
        #Assuming a uniform prior on zL and zS => This doesn't actually do anything as it breaks if it returns -np.inf outside the 
        #prior so I think this will now just always return 0 regardless of the input arguments.
        # prob_4 = dist.Uniform(low=0,high=2).log_prob(zL)+\
        #          dist.Uniform(low=0,high=2).log_prob(zS)
        # data = prob_1+prob_2+prob_3+prob_4
        if likelihood_check:
            prob_1 = dist.Normal(r_obs, sigma_r_obs).log_prob(r_theory)
            prob_2 = dist.Normal(zL_obs, zL_sigma).log_prob(zL)
            prob_3 = dist.Normal(zS_obs, zS_sigma).log_prob(zS)
            prob_4 = MVN_samp(mu_zL_g_L,mu_zS_g_L,
                            sigma_zL_g_L,sigma_zS_g_L,
                            zL,zS, #Should this be zL,zS or zL_obs,zS_obs? Think its ok as zL,zS?
                            sigma_01_g_L,sigma_10_g_L)
            return prob_1+prob_2+prob_3+prob_4
        with numpyro.plate("N", zL_obs.shape[0], subsample_size = subsample_size):
            batch_r_obs = numpyro.subsample(r_obs,event_dim=0)
            batch_sigma_r_obs = numpyro.subsample(sigma_r_obs,event_dim=0)
            batch_r_theory = numpyro.subsample(r_theory,event_dim=0)
            batch_zL_obs = numpyro.subsample(zL_obs,event_dim=0)
            batch_zS_obs = numpyro.subsample(zS_obs,event_dim=0)
            batch_zL_sigma = numpyro.subsample(zL_sigma,event_dim=0)
            batch_zS_sigma = numpyro.subsample(zS_sigma,event_dim=0)
            batch_zL = numpyro.subsample(zL,event_dim=0)
            batch_zS = numpyro.subsample(zS,event_dim=0)
            batch_prob_1 = dist.Normal(batch_r_obs, batch_sigma_r_obs).log_prob(batch_r_theory)
            batch_prob_2 = dist.Normal(batch_zL_obs, batch_zL_sigma).log_prob(batch_zL)
            batch_prob_3 = dist.Normal(batch_zS_obs, batch_zS_sigma).log_prob(batch_zS)
            batch_prob_4 = MVN_samp(mu_zL_g_L,mu_zS_g_L,
                            sigma_zL_g_L,sigma_zS_g_L,
                            batch_zL,batch_zS, #Should this be zL,zS or zL_obs,zS_obs? Think its ok as zL,zS?
                            sigma_01_g_L,sigma_10_g_L)
            batch = batch_prob_1+batch_prob_2+batch_prob_3+batch_prob_4
            L = numpyro.factor("Likelihood",batch)
    elif photometric and contaminated:
        P_tau = P_tau.astype('float') #Needs to be a float for dist.Categorical to work
        print('Assuming contaminated, with photometric redshifts')
        print('NOTE: Need to come up with a test function (e.g. known likelihood) to see what this is actually doing, rather than just\n'+\
            ' assuming that because it gives the right answer it must be correct.')
        mu_zL_g_NL = jnp.squeeze(numpyro.sample("mu_zL_g_NL", dist.Uniform(0,5),sample_shape=(1,),rng_key=key))
        mu_zS_g_NL = jnp.squeeze(numpyro.sample("mu_zS_g_NL", dist.Uniform(0,5),sample_shape=(1,),rng_key=key))
        sigma_zL_g_NL = jnp.squeeze(numpyro.sample("sigma_zL_g_NL", dist.Uniform(0.01,5),sample_shape=(1,),rng_key=key))
        sigma_zS_g_NL = jnp.squeeze(numpyro.sample("sigma_zS_g_NL", dist.Uniform(0.01,5),sample_shape=(1,),rng_key=key))
        if cov_redshift:
            sigma_01_g_NL =  jnp.squeeze(numpyro.sample("sigma_01_g_NL", dist.Uniform(0.01,5),sample_shape=(1,),rng_key=key))
            sigma_10_g_NL =  jnp.squeeze(numpyro.sample("sigma_10_g_NL", dist.Uniform(0.01,5),sample_shape=(1,),rng_key=key))
        else:
            sigma_01_g_NL =  0.0;sigma_10_g_NL =  0.0      
        # dist.Mixture(dist.Categorical(jnp.array([P_tau, 1-P_tau]).T),
                   #         [dist.Normal(r_theory, sigma_r_obs),
                    #         dist.Normal(r_theory, sigma_r_obs_2)]).log_prob(r_obs)
        prob_1a = dist.Normal(r_obs, sigma_r_obs).log_prob(r_theory)
        prob_1b = dist.Normal(r_obs, sigma_r_obs_2).log_prob(r_theory)
        prob_2 = dist.Normal(zL_obs, zL_sigma).log_prob(zL)
        prob_3 = dist.Normal(zS_obs, zS_sigma).log_prob(zS)
        prob_4a = MVN_samp(mu_zL_g_L,mu_zS_g_L,sigma_zL_g_L,sigma_zS_g_L,zL,zS,sigma_01_g_L,sigma_10_g_L)
        prob_4b = MVN_samp(mu_zL_g_NL,mu_zS_g_NL,sigma_zL_g_NL,sigma_zS_g_NL,zL,zS,sigma_01_g_NL,sigma_10_g_NL)
        '''
        Seems to be a problem with very small numbers - can cope if I increase the precision but still with only very small numbers of 
        systems => Problem fixed by having P_tau!=1.0 (even 0.9 fixed it).
        '''
        #'log_prob' finds the natural logarithm (not log10), hence these are natural-logged:
        L_1 =  likelihood_PC(P_tau,prob_1a,prob_1b,prob_2,prob_3,prob_4a,prob_4b)
        L = numpyro.factor("Likelihood",L_1)
    else:
        print('Assuming not contaminated, with spectroscopic redshifts')
        assert not photometric and not contaminated
        with numpyro.plate("N", r_theory.shape[0], subsample_size=subsample_size):
            batch_r_obs = numpyro.subsample(r_obs,event_dim=0)
            batch_r_theory = numpyro.subsample(r_theory,event_dim=0)
            batch_sigma_r_obs = numpyro.subsample(sigma_r_obs,event_dim=0)
            numpyro.sample("r",dist.Normal(batch_r_theory, batch_sigma_r_obs), obs=batch_r_obs)
        # numpyro.sample("r", dist.Normal(r_theory, sigma_r_obs), obs=r_obs)


def run_MCMC(photometric,contaminated,cosmo_type,
            zL_obs,zS_obs,sigma_zL_obs,sigma_zS_obs,
            r_obs,sigma_r_obs,sigma_r_obs_2,P_tau,
            num_warmup = 200,num_samples=1000,num_chains=2,
            H0=np.nan,target_accept_prob=0.8,cov_redshift=False,warmup_file=np.nan):
    model_args = {'zL_obs':zL_obs,'zS_obs':zS_obs,
                'sigma_zL_obs':sigma_zL_obs,'sigma_zS_obs':sigma_zS_obs,
                'r_obs':r_obs,'sigma_r_obs':sigma_r_obs,'sigma_r_obs_2':sigma_r_obs_2,
                'P_tau':P_tau,'cosmo_type':cosmo_type,
                'photometric':photometric,'contaminated':contaminated,
                'H0':H0,'cov_redshift':cov_redshift}
    key = jax.random.PRNGKey(0)
    print(f'Target Accept Prob: {target_accept_prob}')
    st = time.time()
    j_likelihood_SL(**model_args,key=key,early_return=True)
    mt=time.time()
    j_likelihood_SL(**model_args,key=key,early_return=True)
    et=time.time()
    print('Uncompiled time',mt-st)
    print('Compiled time',et-mt)
    #USEFUL LINK REGARDING SPEEDING UP NUTS AND HMCECS:
    #https://forum.pyro.ai/t/scalability-of-hmcecs/5349/12
    inner_kernel = NUTS(model = j_likelihood_SL,target_accept_prob = target_accept_prob)
    outer_kernel = HMCECS(inner_kernel, num_blocks=100)
    sampler_0 = MCMC(outer_kernel,
                    num_warmup=num_warmup,
                    num_samples=num_samples,
                    num_chains=num_chains,
                    progress_bar=True)
    sampler_0.warmup(key,**model_args,collect_warmup=True)
    ##
    warmup_dict = JAX_samples_to_dict(sampler_0,separate_keys=True,cosmo_type=cosmo_type)
    db_JAX_warmup = pd.DataFrame(warmup_dict)
    db_JAX_warmup.to_csv(warmup_file,index=False)
    ##
    sampler_0.run(key,**model_args,key=None)
    return sampler_0
    # Without HMCECS:
    # sampler_0 = infer.MCMC(
    #     infer.NUTS(model = j_likelihood_SL,
    #                target_accept_prob = target_accept_prob),
    #     num_warmup=num_warmup,
    #     num_samples=num_samples,
    #     num_chains=num_chains,
    #     progress_bar=True)
    return sampler_0

'''
A note about the cosmology checks: Need to check if some of the values I am checking are physical (e.g. can Ode be >1?). If not, then
its not really a problem that it fails these checks. 
'''
print('Importing Omega_k!=0 package') #Is ok to use for parameter-searching for Omega-k as the probability of getting Ok==0 exactly is infinitely small
from cosmology_JAX import j_r_SL
print('Importing Omega_k==0 package')
from cosmology_JAX_flat import j_r_SL as j_r_SL_flat

def Omega_k_check():
    print('Running Omega_k check')
    '''
    Checks that there are no differences in the cosmology_JAX and cosmology_JAX_flat packages, except for the couple of lines
    where the curvature alters the function. This checks up to the cosmo_check function (assuming there is no cosmo functions 
    defined beyond this)
    '''
    lines_0=[];lines_f=[]
    with open('./cosmology_JAX.py','r') as c_0:
        for line_i in c_0: lines_0.append(line_i)
    with open('./cosmology_JAX_flat.py','r') as c_f:
        for line_f in c_f: lines_f.append(line_f)
    for ii in tqdm(range(len(lines_0))):
        line_0_ii = lines_0[ii].strip().replace('\n','')
        line_f_ii = lines_f[ii].strip().replace('\n','')
        assert (line_0_ii==line_f_ii) or\
                ((line_0_ii=='from jax_cosmo import background') and\
                 (line_f_ii=='from jax_cosmo import background_flat as background')) or\
                ((line_0_ii=='j_D_LS_ii = jnp.where((j_cosmo.k==-1).T,a3*b3*c3,jnp.where((j_cosmo.k==1).T,a1*b1*c1,a2*b2))' and\
                 (line_f_ii=='j_D_LS_ii = a2*b2')))
        if line_0_ii=='def cosmo_check():' and (line_0_ii==line_f_ii): break

Omega_k_check() #Do not remove this!
