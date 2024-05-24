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
#import arviz as az
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
def likelihood_PC(P_tau,prob_1a,prob_1b,prob_2,prob_3,prob_4a,prob_4b,prob_5a,prob_5b):
        return (jnp.log(P_tau*jnp.exp(prob_1a)*jnp.exp(prob_4a)*jnp.exp(prob_5a)+\
                       (1-P_tau)*jnp.exp(prob_1b)*jnp.exp(prob_4b)*jnp.exp(prob_5b))+prob_2+prob_3)

def j_likelihood_SL(zL_obs,zS_obs,sigma_zL_obs,sigma_zS_obs,r_obs,sigma_r_obs,sigma_r_obs_2=[np.nan],P_tau = [],cosmo_type='',
                    photometric=False,contaminated=False,H0=np.nan,key=None,
                    likelihood_check=False,likelihood_dict = {},cov_redshift=False,early_return=False,
                    batch_bool=True, wa_const = False, w0_const = False,GMM_zL = False,GMM_zS = False):
    s8 = 0.8;n_s = 0.96;Ob=0; #Putting all the matter in dark-matter (doesn't make a difference)
    if len(zL_obs)<10000: subsample_size = 8000
    else: subsample_size = 12000
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
            #assert False #Think these should all be uniform distributions (at least zL, zS), as marginalising over them here???? Then the normal distribution appears in the likelihood term.
            # zL = numpyro.sample('zL',dist.TruncatedNormal(jnp.array(zL_obs),zL_sigma,low=0),sample_shape=(1,),rng_key=key).flatten()
            # zS = numpyro.sample('zS',dist.TruncatedNormal(jnp.array(zS_obs),zS_sigma,low=0),sample_shape=(1,),rng_key=key).flatten()
            #
            zL_obs_low_lim = jnp.array(zL_obs-3*zL_sigma)
            zL_obs_low_lim = zL_obs_low_lim*(zL_obs_low_lim>0) #Minimum value is 0
            zL_obs_up_lim = jnp.array(zL_obs+3*zL_sigma)
            zL = numpyro.sample('zL',dist.Uniform(low = zL_obs_low_lim,high = zL_obs_up_lim),sample_shape=(1,),rng_key=key).flatten()
            ##Minimum value is zL. First term: zS if zS>0, else 0. Second term, zL if zS<zL, else 0
            zS_obs_low_lim = jnp.array(zS_obs-3*zS_sigma)
            #CHANGING THIS TO 0 - NO LONGER ASSERTING ZL<ZS HERE:
            # zS_obs_low_lim = zS_obs_low_lim*(zS_obs_low_lim>zL) + zL*(zS_obs_low_lim<zL)
            zS_obs_low_lim = zS_obs_low_lim*(zS_obs_low_lim>0)
            zS_obs_up_lim = jnp.array(zS_obs+3*zS_sigma)
            #CHANGING THIS TO 0 - NO LONGER ASSERTING ZL<ZS HERE:
            zS = numpyro.sample('zS',dist.Uniform(low = zS_obs_low_lim, high = zS_obs_up_lim),sample_shape=(1,),rng_key=key).flatten()
            #
            if GMM_zL:
                mu_zL_g_L_A = jnp.squeeze(numpyro.sample("mu_zL_g_L_A", dist.Uniform(0.1,1.5),sample_shape=(1,),rng_key=key))
                sigma_zL_g_L_A = jnp.squeeze(numpyro.sample("sigma_zL_g_L_A", dist.Uniform(0.01,1),sample_shape=(1,),rng_key=key))
                mu_zL_g_L_B = jnp.squeeze(numpyro.sample("mu_zL_g_L_B", dist.Uniform(0.1,1.5),sample_shape=(1,),rng_key=key))
                sigma_zL_g_L_B = jnp.squeeze(numpyro.sample("sigma_zL_g_L_B", dist.Uniform(0.01,1),sample_shape=(1,),rng_key=key))
                w_zL = numpyro.sample('w_zL', dist.Uniform(0,1),rng_key=key)
            else:
                mu_zL_g_L = jnp.squeeze(numpyro.sample("mu_zL_g_L", dist.Uniform(0.1,1.5),sample_shape=(1,),rng_key=key))
                sigma_zL_g_L = jnp.squeeze(numpyro.sample("sigma_zL_g_L", dist.Uniform(0.01,1),sample_shape=(1,),rng_key=key))
            if GMM_zS:
                mu_zS_g_L_A = jnp.squeeze(numpyro.sample("mu_zS_g_L_A", dist.Uniform(0.1,2),sample_shape=(1,),rng_key=key))
                sigma_zS_g_L_A = jnp.squeeze(numpyro.sample("sigma_zS_g_L_A", dist.Uniform(0.01,1.5),sample_shape=(1,),rng_key=key))
                mu_zS_g_L_B = jnp.squeeze(numpyro.sample("mu_zS_g_L_B", dist.Uniform(0.1,2),sample_shape=(1,),rng_key=key))
                sigma_zS_g_L_B = jnp.squeeze(numpyro.sample("sigma_zS_g_L_B", dist.Uniform(0.01,1.5),sample_shape=(1,),rng_key=key))
                w_zS = numpyro.sample('w_zS', dist.Uniform(0,1),rng_key=key)
            else:
                mu_zS_g_L = jnp.squeeze(numpyro.sample("mu_zS_g_L", dist.Uniform(0.1,2),sample_shape=(1,),rng_key=key))
                sigma_zS_g_L = jnp.squeeze(numpyro.sample("sigma_zS_g_L", dist.Uniform(0.01,1.5),sample_shape=(1,),rng_key=key))
            if cov_redshift:
                assert not GMM_zL and not GMM_zS
                sigma_01_g_L =  jnp.squeeze(numpyro.sample("sigma_01_g_L", dist.Uniform(0.01,2),sample_shape=(1,),rng_key=key))
                sigma_10_g_L =  jnp.squeeze(numpyro.sample("sigma_10_g_L", dist.Uniform(0.01,2),sample_shape=(1,),rng_key=key))
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
            # Sampling OM and Ok uniformly. Ode is therefore sampled from a weird shaped distribution in the range (-1,2), but
            # which is uniform in the range (0,1)
            # This can therefore be cropped to a uniform distribution by setting the likelihood to -np.inf if Ode is outside of (0,1).
            Ok = jnp.squeeze(numpyro.sample("Ok", dist.Uniform(-1,1),sample_shape=(1,),rng_key=key))
            Ode = numpyro.deterministic('Ode',1-(OM+Ok))
        if cosmo_type in ['LambdaCDM','FlatLambdaCDM']:
            print('Assuming universe has a cosmological constant')
            w = numpyro.deterministic('w',-1.0)
            wa = numpyro.deterministic('wa',0.0)
        elif wa_const == True and w0_const == False:
            print('Assuming a non-evolving dark energy equation of state')
            w = jnp.squeeze(numpyro.sample("w", dist.Uniform(-6,4),sample_shape=(1,),rng_key=key)) #Physicality constraints, (-6,4)
            wa = numpyro.deterministic('wa',0.0)
        elif wa_const == False and w0_const == True:
            print('Assuming an evolving dark energy equation of state, but with w0 fixed at -1.')
            w = numpyro.deterministic('w',-1.0)
            wa = jnp.squeeze(numpyro.sample("wa", dist.Uniform(-3,1),sample_shape=(1,),rng_key=key)) #Matching Tian's constraints for now
        else:
            print('Assuming non-trivial dark energy equation of state')
            w = jnp.squeeze(numpyro.sample("w", dist.Uniform(-6,4),sample_shape=(1,),rng_key=key)) #Physicality constraints, (-6,4)
            wa = jnp.squeeze(numpyro.sample("wa", dist.Uniform(-3,1),sample_shape=(1,),rng_key=key)) #Matching Tian's constraints for now
    if likelihood_check:
            OM = likelihood_dict['OM'];Ok = likelihood_dict['Ok']
            w = likelihood_dict['w'];wa=likelihood_dict['wa'];
            r_obs =likelihood_dict['r_obs'];sigma_r_obs = likelihood_dict['sigma_r_obs'];
            zL = likelihood_dict['zL'];zS = likelihood_dict['zS']
            zL_obs = likelihood_dict['zL_obs'];zS_obs = likelihood_dict['zS_obs']
            try:
                zL_sigma = likelihood_dict['zL_sigma'];zS_sigma = likelihood_dict['zS_sigma']
            except: pass
            try:
                mu_zL_g_L,mu_zS_g_L = likelihood_dict['mu_zL_g_L'],likelihood_dict['mu_zS_g_L']
                sigma_zL_g_L,sigma_zS_g_L = likelihood_dict['sigma_zL_g_L'],likelihood_dict['sigma_zS_g_L']
                sigma_01_g_L,sigma_10_g_L = likelihood_dict['sigma_01_g_L'],likelihood_dict['sigma_10_g_L']
            except: pass
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
                            [dist.TruncatedNormal(batch_r_theory, batch_sigma_r_obs,low=0),
                             dist.TruncatedNormal(batch_r_theory, batch_sigma_r_obs_2,low=0)]).log_prob(batch_r_obs)
            batch_prob_1 = jnp.where(batch_r_obs<0,-np.inf,batch_prob_1)
            batch_prob_1 = jnp.where(Ode*jnp.ones(len(batch_prob_1))<0,-np.inf,batch_prob_1)
            batch_prob_1 = jnp.where(Ode*jnp.ones(len(batch_prob_1))>1,-np.inf,batch_prob_1)
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
        if batch_bool:
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
                batch_prob_1 = dist.TruncatedNormal(batch_r_theory, batch_sigma_r_obs,low=0).log_prob(batch_r_obs)
                batch_prob_1 = jnp.where(batch_r_obs<0,-np.inf,batch_prob_1)
                batch_prob_2 = dist.TruncatedNormal(batch_zL, batch_zL_sigma,low = 0).log_prob(batch_zL_obs)
                batch_prob_2 = jnp.where(batch_zL_obs<0,-np.inf,batch_prob_2)
                batch_prob_3 = dist.TruncatedNormal(batch_zS,batch_zS_sigma, low = 0).log_prob(batch_zS_obs)
                batch_prob_3 = jnp.where(batch_zS_obs<0,-np.inf,batch_prob_3)
                if GMM_zL:
                    batch_prob_4 = dist.Mixture(dist.Categorical(probs=jnp.array([w_zL,1-w_zL])),
                            [dist.TruncatedNormal(mu_zL_g_L_A, sigma_zL_g_L_A,low=0),
                             dist.TruncatedNormal(mu_zL_g_L_B, sigma_zL_g_L_B,low=0)]).log_prob(batch_zL)+\
                        dist.Normal(batch_zL, batch_zL_sigma).log_prob(batch_zL_obs)
                else:
                    batch_prob_4 = dist.TruncatedNormal(mu_zL_g_L, sigma_zL_g_L, low = 0).log_prob(batch_zL)
                    batch_prob_4 = jnp.where(batch_zL<0,-np.inf,batch_prob_4)
                if GMM_zS:
                    #CHANGING THIS TO 0 - NO LONGER ASSERTING ZL<ZS HERE:
                    batch_prob_5 = dist.Mixture(dist.Categorical(probs=jnp.array([w_zS,1-w_zS])),
                            [dist.TruncatedNormal(mu_zS_g_L_A, sigma_zS_g_L_A, low=0),
                             dist.TruncatedNormal(mu_zS_g_L_B, sigma_zS_g_L_B, low=0)]).log_prob(batch_zS)+\
                        dist.Normal(batch_zS, batch_zS_sigma).log_prob(batch_zS_obs)
                else:
                    #CHANGING THIS TO 0 - NO LONGER ASSERTING ZL<ZS HERE:
                    batch_prob_5 = dist.TruncatedNormal(mu_zS_g_L, sigma_zS_g_L, low = 0).log_prob(batch_zS)
                    #COMMENTING OUT THIS - NO LONGER ASSERTING ZL<ZS HERE:
                    # batch_prob_5 = jnp.where(batch_zS<batch_zL,-np.inf,batch_prob_5)
                batch = batch_prob_1+batch_prob_2+batch_prob_3+batch_prob_4+batch_prob_5
                batch = jnp.where(Ode*jnp.ones(len(batch))<0,-np.inf,batch)
                batch = jnp.where(Ode*jnp.ones(len(batch))>1,-np.inf,batch)
                # jax.debug.print('BATCH LIKELIHOOD {a},{b},{c},{d},{e},{f},{g}',
                #                 a=batch_prob_1,b=batch_prob_2,c=batch_prob_3,d=batch_prob_4,e=batch_prob_5,f=batch,g=len(batch))
                # print('LIKELIHOOD END')

                #CHANGING THIS TO INF no longer - not ASSERTING ZL<ZS IN LIKELIHOOD:
                # batch = jnp.where(batch_zS<batch_zL,-np.inf,batch)
                L = numpyro.factor("Likelihood",batch)

        else:
            prob_1 = dist.TruncatedNormal(r_theory, sigma_r_obs,low=0).log_prob(r_obs)
            prob_1 = jnp.where(r_obs<0,-np.inf,prob_1)
            prob_2 = dist.TruncatedNormal(zL, zL_sigma,low = 0).log_prob(zL_obs)
            prob_2 = jnp.where(zL_obs<0,-np.inf,prob_2)
            prob_3 = dist.TruncatedNormal(zS, zS_sigma, low = 0).log_prob(zS_obs)
            prob_3 = jnp.where(zS_obs<0,-np.inf,prob_3)
            if GMM_zL:
                prob_4 = dist.Mixture(dist.Categorical(probs=jnp.array([w_zL,1-w_zL])),
                            [dist.TruncatedNormal(mu_zL_g_L_A, sigma_zL_g_L_A,low=0),
                             dist.TruncatedNormal(mu_zL_g_L_B, sigma_zL_g_L_B,low=0)]).log_prob(zL)+\
                        dist.Normal(zL, zL_sigma).log_prob(zL_obs)
            else:
                prob_4 = dist.TruncatedNormal(mu_zL_g_L, sigma_zL_g_L, low = 0).log_prob(zL)
                prob_4 = jnp.where(zL<0,-np.inf,prob_4)
            if GMM_zS:
                # CHANGING THIS TO 0 - NO LONGER ASSERTING ZL<ZS HERE:
                prob_5 = dist.Mixture(dist.Categorical(probs=jnp.array([w_zS,1-w_zS])),
                            [dist.TruncatedNormal(mu_zS_g_L_A, sigma_zS_g_L_A,low=0),
                             dist.TruncatedNormal(mu_zS_g_L_B, sigma_zS_g_L_B,low=0)]).log_prob(zS)+\
                         dist.Normal(zS, zS_sigma).log_prob(zS_obs)
            else:
                # CHANGING THIS TO 0 - NO LONGER ASSERTING ZL<ZS HERE:
                prob_5 = dist.TruncatedNormal(mu_zS_g_L, sigma_zS_g_L, low = 0).log_prob(zS)
                # COMMENTING OUT THIS - NO LONGER ASSERTING ZL<ZS HERE:
                # prob_5 = jnp.where(zS<zL,-np.inf,prob_5)
            # prob_4 = MVN_samp(mu_zL_g_L,mu_zS_g_L,
            #                 sigma_zL_g_L,sigma_zS_g_L,
            #                 zL,zS, #Should this be zL,zS or zL_obs,zS_obs? Think its ok as zL,zS?
            #                 sigma_01_g_L,sigma_10_g_L)
            prob = prob_1+prob_2+prob_3+prob_4+prob_5
            prob = jnp.where(Ode*jnp.ones(len(prob))<0,-np.inf,prob)
            prob = jnp.where(Ode*jnp.ones(len(prob))>1,-np.inf,prob)
            # print('LIKELIHOOD END')
            #CHANGING THIS TO INF - ASSERTING ZL<ZS IN LIKELIHOOD:
            # prob = jnp.where(zS<zL,-np.inf,prob)
            # jax.debug.print('LIKELIHOOD comp {a},{b},{c},{d},{e},{f}',a=prob_1,b=prob_2,c=prob_3,d=prob_4,e=prob_4,f=prob_5)
            # jax.debug.print('LIKELIHOOD {a},{b},{c},{d},{e}',a=jnp.max(prob),b=jnp.min(prob),c=jnp.sum(jnp.isinf(prob)),d=prob,e=len(prob))
            if likelihood_check: return prob
            L = numpyro.factor("Likelihood",prob)

    elif photometric and contaminated:
        P_tau = P_tau.astype('float') #Needs to be a float for dist.Categorical to work
        print('Assuming contaminated, with photometric redshifts')
        print('NOTE: Need to come up with a test function (e.g. known likelihood) to see what this is actually doing, rather than just\n'+\
            ' assuming that because it gives the right answer it must be correct.')
        if GMM_zL:
            assert False #Not yet implemented
        else:
            mu_zL_g_NL = jnp.squeeze(numpyro.sample("mu_zL_g_NL", dist.Uniform(0,5),sample_shape=(1,),rng_key=key))
            sigma_zL_g_NL = jnp.squeeze(numpyro.sample("sigma_zL_g_NL", dist.Uniform(0.01,5),sample_shape=(1,),rng_key=key))
        if GMM_zS:
            assert False #Not yet implemented
        else:
            mu_zS_g_NL = jnp.squeeze(numpyro.sample("mu_zS_g_NL", dist.Uniform(0,5),sample_shape=(1,),rng_key=key))
            sigma_zS_g_NL = jnp.squeeze(numpyro.sample("sigma_zS_g_NL", dist.Uniform(0.01,5),sample_shape=(1,),rng_key=key))
        if cov_redshift:
            assert not GMM_zL and not GMM_zS
            sigma_01_g_NL =  jnp.squeeze(numpyro.sample("sigma_01_g_NL", dist.Uniform(0.01,5),sample_shape=(1,),rng_key=key))
            sigma_10_g_NL =  jnp.squeeze(numpyro.sample("sigma_10_g_NL", dist.Uniform(0.01,5),sample_shape=(1,),rng_key=key))
        else:
            sigma_01_g_NL =  0.0;sigma_10_g_NL =  0.0      
        # dist.Mixture(dist.Categorical(jnp.array([P_tau, 1-P_tau]).T),
                   #         [dist.Normal(r_theory, sigma_r_obs),
                    #         dist.Normal(r_theory, sigma_r_obs_2)]).log_prob(r_obs)
        prob_1a = dist.TruncatedNormal(r_theory, sigma_r_obs,low = 0).log_prob(r_obs)
        prob_1a = jnp.where(r_obs<0,-np.inf,prob_1a)
        prob_1b = dist.TruncatedNormal(r_theory, sigma_r_obs_2, low = 0).log_prob(r_obs)
        prob_1b = jnp.where(r_obs<0,-np.inf,prob_1b)
        prob_2 = dist.TruncatedNormal(zL, zL_sigma, low = 0).log_prob(zL_obs)
        prob_2 = jnp.where(zL_obs<0,-np.inf,prob_2)
        prob_3 = dist.TruncatedNormal(zS, zS_sigma, low = 0).log_prob(zS_obs)
        prob_3 = jnp.where(zS_obs<0,-np.inf,prob_3)
        prob_4a = dist.TruncatedNormal(mu_zL_g_L,sigma_zL_g_L,low=0).log_prob(zL)
        prob_4a = jnp.where(zL<0,-np.inf,prob_4a)
        prob_5a = dist.TruncatedNormal(mu_zS_g_L,sigma_zS_g_L,low=0).log_prob(zS)
        prob_5a = jnp.where(zS<0,-np.inf,prob_5a)
        if GMM_zL:
            assert False #Not yet implemented
        else:
            prob_4b = dist.TruncatedNormal(mu_zL_g_NL,sigma_zL_g_NL,low=0).log_prob(zL)
            prob_4b = jnp.where(zL<0,-np.inf,prob_4b)
        if GMM_zS:
            assert False #Not yet implemented
        else:
            #QUICK CHECK
            prob_5b = dist.TruncatedNormal(mu_zS_g_NL,sigma_zS_g_NL,low=zL).log_prob(zS)
            prob_5b = jnp.where(zS<zL,-np.inf,prob_5b)
        # prob_4a = MVN_samp(mu_zL_g_L,mu_zS_g_L,sigma_zL_g_L,sigma_zS_g_L,zL,zS,sigma_01_g_L,sigma_10_g_L)
        # prob_4b = MVN_samp(mu_zL_g_NL,mu_zS_g_NL,sigma_zL_g_NL,sigma_zS_g_NL,zL,zS,sigma_01_g_NL,sigma_10_g_NL)
        '''
        Seems to be a problem with very small numbers - can cope if I increase the precision but still with only very small numbers of 
        systems => Problem fixed by having P_tau!=1.0 (even 0.9 fixed it).
        '''
        #'log_prob' finds the natural logarithm (not log10), hence these are natural-logged:
        prob =  likelihood_PC(P_tau,prob_1a,prob_1b,prob_2,prob_3,prob_4a,prob_4b,prob_5a,prob_5b)
        prob = jnp.where(Ode*jnp.ones(len(prob))<0,-np.inf,prob)
        prob = jnp.where(Ode*jnp.ones(len(prob))>1,-np.inf,prob)
        L = numpyro.factor("Likelihood",prob)
    else:
        print('Assuming not contaminated, with spectroscopic redshifts')
        assert not photometric and not contaminated
        if batch_bool:
            with numpyro.plate("N", r_theory.shape[0], subsample_size=subsample_size):
                batch_r_obs = numpyro.subsample(r_obs,event_dim=0)
                batch_r_theory = numpyro.subsample(r_theory,event_dim=0)
                batch_sigma_r_obs = numpyro.subsample(sigma_r_obs,event_dim=0)
                batch_prob = dist.TruncatedNormal(batch_r_theory, batch_sigma_r_obs, low = 0).log_prob(batch_r_obs)
                batch_prob = jnp.where(batch_r_obs<0,-np.inf,batch_prob)
                batch_prob = jnp.where(Ode*jnp.ones(len(batch_prob))<0,-np.inf,batch_prob)
                batch_prob = jnp.where(Ode*jnp.ones(len(batch_prob))>1,-np.inf,batch_prob)
                L = numpyro.factor('Likelihood',batch_prob)
        else:
                print('Not using batching')
                prob = dist.TruncatedNormal(r_theory, sigma_r_obs, low = 0).log_prob(r_obs)
                prob = jnp.where(r_obs<0,-np.inf,prob)
                prob = jnp.where(Ode*jnp.ones(len(prob))<0,-np.inf,prob)
                prob = jnp.where(Ode*jnp.ones(len(prob))>1,-np.inf,prob)
                if likelihood_check: 
                    if np.isnan(np.array(L)).any() or np.isinf(np.array(L)).any():
                        print('Infinite or nan likelihood:',L,r_theory,sigma_r_obs,r_obs,cosmo,zL,zS)
                        print('OM, Ok:',OM, Ok)
                    return L
                numpyro.factor('Likelihood',prob)
        # numpyro.sample("r", dist.Normal(r_theory, sigma_r_obs), obs=r_obs)
        

def run_MCMC(photometric,contaminated,cosmo_type,
            zL_obs,zS_obs,sigma_zL_obs,sigma_zS_obs,
            r_obs,sigma_r_obs,sigma_r_obs_2,P_tau,
            num_warmup = 200,num_samples=1000,num_chains=2,
            H0=np.nan,target_accept_prob=0.8,cov_redshift=False,warmup_file=np.nan,
            batch_bool=True,wa_const=False,w0_const=False,GMM_zL = False,GMM_zS = False,key_int = 0):
    print('Random key:',key_int)
    model_args = {'zL_obs':zL_obs,'zS_obs':zS_obs,
                'sigma_zL_obs':sigma_zL_obs,'sigma_zS_obs':sigma_zS_obs,
                'r_obs':r_obs,'sigma_r_obs':sigma_r_obs,'sigma_r_obs_2':sigma_r_obs_2,
                'P_tau':P_tau,'cosmo_type':cosmo_type,
                'photometric':photometric,'contaminated':contaminated,
                'H0':H0,'cov_redshift':cov_redshift,'batch_bool':batch_bool,
                'wa_const':wa_const,'w0_const':w0_const,'GMM_zL':GMM_zL,'GMM_zS':GMM_zS}
    key = jax.random.PRNGKey(key_int)
    print(f'Target Accept Prob: {target_accept_prob}')
    print(f'Batch bool: {batch_bool}')
    st = time.time()
    j_likelihood_SL(**model_args,key=key,early_return=True)
    mt=time.time()
    j_likelihood_SL(**model_args,key=key,early_return=True)
    et=time.time()
    print('Uncompiled time',mt-st)
    print('Compiled time',et-mt)
    #USEFUL LINK REGARDING SPEEDING UP NUTS AND HMCECS:
    #https://forum.pyro.ai/t/scalability-of-hmcecs/5349/12
    if batch_bool:
        inner_kernel = NUTS(model = j_likelihood_SL,target_accept_prob = target_accept_prob)
        outer_kernel = HMCECS(inner_kernel, num_blocks=100)
    else:
        outer_kernel =  NUTS(model = j_likelihood_SL,target_accept_prob = target_accept_prob)
    sampler_0 = MCMC(outer_kernel,
                    num_warmup=num_warmup,
                    num_samples=num_samples,
                    num_chains=num_chains,
                    progress_bar=True)
    sampler_0.warmup(key,**model_args,collect_warmup=True)
    ##
    warmup_dict = JAX_samples_to_dict(sampler_0,separate_keys=True,cosmo_type=cosmo_type,wa_const=wa_const,w0_const=w0_const)
    db_JAX_warmup = pd.DataFrame(warmup_dict)
    db_JAX_warmup.to_csv(warmup_file,index=False)
    print(f'Saved warmup to {warmup_file}')
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
