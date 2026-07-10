'''
Code description: This file infers the cosmological posteriors (as well as those for various population hyperparameters) given a sample of impure and/or photometric
strong lenses.
The MCMC is run using NUTS (https://num.pyro.ai/en/latest/mcmc.html#numpyro.infer.hmc.NUTS).
'''

from astropy.cosmology import LambdaCDM,FlatLambdaCDM,wCDM,FlatwCDM,w0waCDM
from numpyro import distributions as dist,infer
from numpyro.infer import MCMC,NUTS,HMC,HMCECS,init_to_value,init_to_uniform
from jax import random,grad, jit
import jax.numpy as jnp
import jax_cosmo as jc
from tqdm import tqdm
import scipy.sparse
import pandas as pd
import numpy as np
import numpyro
import emcee
import sys
import jax
import time
from jax.scipy.stats import truncnorm as jax_truncnorm
from Beta_Distribution_Class import beta_class
from LogNormal_Distribution_Class import jax_lognormal 
from cosmology_JAX_public import j_r_SL
from cosmology_JAX_flat_public import j_r_SL as j_r_SL_flat

def truncnorm_mixture(weights,loc,scale,obs):
    return jnp.log(jnp.sum(jnp.array([weights[i]*jax_truncnorm.pdf(loc=loc[i],
                                                                   scale=scale[i],
                                                                   a=-loc[i]/scale[i],
                                                                   b=jnp.inf,x=obs) 
            for i in range(len(loc))]),axis=0))

@jit
def likelihood_phot_contam(P_tau,prob_r_obs_TP,prob_r_obs_FP,prob_2,prob_3,prob_zL_zS,
                            prob_zL_FP,prob_zS_FP,
                            prob_zL_TP,prob_zS_TP,
                            delta_zL,delta_zS,
                            trapezium_factor_zL,trapezium_factor_zS,redshift_bool,trapezium_factor,prob_f=0.0,
                            ): #prob_zL_zS is not logged
        """
        This is the likelihood function used for a contaminated, photometric population.
        """
        likelihood_part_1_integral = 0.25*delta_zL*delta_zS*jnp.sum(jnp.sum(trapezium_factor*jnp.exp(prob_r_obs_TP+prob_2+prob_3+prob_zL_TP+prob_zS_TP)*prob_zL_zS,axis=1),axis=0)
        likelihood_part_1 = P_tau*likelihood_part_1_integral*jnp.exp(prob_f)
        likelihood_part_2 = (1-P_tau)*jnp.exp(prob_r_obs_FP+prob_zL_FP+prob_zS_FP)
        total_likelihood = jnp.log(likelihood_part_1+likelihood_part_2)
        #When part_1 is zero, can just use the logged version of part_2 alone, and vice versa.
        total_likelihood = jnp.where(likelihood_part_1==0,jnp.log(1-P_tau) + prob_r_obs_FP + prob_zL_FP + prob_zS_FP,total_likelihood)
        total_likelihood = jnp.where(likelihood_part_2==0,jnp.log(P_tau*likelihood_part_1_integral)+prob_f,total_likelihood)
        return total_likelihood

def calculate_integral_variables(zL_obs,zS_obs,zL_sigma,zS_sigma,N_sigma=3,N_zL_step = 11,N_zS_step = 11):
    """
    This function integrates over the redshift uncertainty, via the trapezium rule.
    """
    zL_obs_low_lim = jnp.array(zL_obs-N_sigma*zL_sigma)
    zL_obs_low_lim = zL_obs_low_lim*(zL_obs_low_lim>0) #Minimum value is 0
    zL_obs_up_lim = jnp.array(zL_obs+N_sigma*zL_sigma)
    ##Minimum value is zL. First term: zS if zS>0, else 0. Second term, zL if zS<zL, else 0
    zS_obs_low_lim = jnp.array(zS_obs-N_sigma*zS_sigma)
    zS_obs_low_lim = zS_obs_low_lim*(zS_obs_low_lim>0)
    zS_obs_up_lim = jnp.array(zS_obs+N_sigma*zS_sigma)
    zL_array = jnp.linspace(zL_obs_low_lim,zL_obs_up_lim,N_zL_step)
    zS_array = jnp.linspace(zS_obs_low_lim,zS_obs_up_lim,N_zS_step)
    delta_zL = zL_array[1]-zL_array[0]
    delta_zS = zS_array[1]-zS_array[0]
    trapezium_factor_zL = jnp.array([1]+[2]*int(N_zL_step-2)+[1])
    trapezium_factor_zS = jnp.array([1]+[2]*int(N_zS_step-2)+[1])
    delta_z_error = 0.01
    redshift_bool = jnp.array([[zL_array[elem_zL]<(zS_array[elem_zS]-delta_z_error) for elem_zS in range(len(zS_array))] for elem_zL in range(len(zL_array))])
    # Swap round lens and source if zL>zS - these are then subsequently massively downweighted in the likelihood calculation. 
    adjusted_zL_array = jnp.array([[jnp.where(zL_array[elem_zL]<(zS_array[elem_zS]-delta_z_error),zL_array[elem_zL],zS_array[elem_zS]) for elem_zS in range(len(zS_array))] for elem_zL in range(len(zL_array))])
    adjusted_zS_array = jnp.array([[jnp.where(zL_array[elem_zL]<(zS_array[elem_zS]-delta_z_error),zS_array[elem_zS],zL_array[elem_zL]+0.1) for elem_zS in range(len(zS_array))] for elem_zL in range(len(zL_array))])
    trapezium_factor = ((trapezium_factor_zL*jnp.array([trapezium_factor_zS]).T).T)[:,:,jnp.newaxis]
    z_diff = jnp.array([[adjusted_zS_array[elem_zL][elem_zS]-adjusted_zL_array[elem_zL][elem_zS] \
                                                                         for elem_zS in range(len(zS_array))] \
                                                                         for elem_zL in range(len(zL_array))])
    return zL_array,zS_array,delta_zL,delta_zS,trapezium_factor_zL,trapezium_factor_zS,redshift_bool,\
            adjusted_zL_array,adjusted_zS_array,trapezium_factor,z_diff

def j_likelihood_SL(#See comments below for parameter definitions.
                    zL_obs,zS_obs,sigma_zL_obs,sigma_zS_obs,r_obs,sigma_r_obs,P_tau_0 = [],cosmo_type='',
                    photometric=False,contaminated=False,H0=np.nan,key=None,
                    early_return=False,
                    wa_const = False, w0_const = False,
                    use_true_z_phot_code=False,zL_true=[],zS_true=[],
                    fixed_alpha=False,alpha_dict = {},
                    fixed_beta_gamma=False,beta_dict = {},gamma_dict = {},
                    beta_gamma_lens=False,fixed_beta_gamma_lens = False, beta_lens_dict = {},gamma_lens_dict = {},
                    remove_gamma_lens = False,z_diff = None,
                    s_dict = {},fixed_s = False,
                    #The arguments below are used internally in the code to help marginalise over the redshift uncertainty, and are not required to be specified seperately.
                    #This is done so this part of the code can be run once, rather than for every MCMC step.
                    zL_array=None,zS_array=None,delta_zL=None,delta_zS=None,trapezium_factor_zL=None,trapezium_factor_zS=None,
                    redshift_bool=None,adjusted_zL_array=None,adjusted_zS_array=None,prob_2=None,prob_3 = None,
                    trapezium_factor=None,prob_zL_zS = None,
                    ):
    '''
    Main input args:
    zL_obs: Observed lens redshift
    zS_obs: Observed source redshift
    sigma_zL_obs: Observed lens redshift uncertainty
    sigma_zS_obs: Observed source redshift uncertainty
    r_obs: Observed 'r' ratio (= c^2 * theta_E/ (4pi * sigma_v^2), where theta_E = Einstein radius, sigma_v = velocity dispersion).
    sigma_r_obs: Observed r ratio uncertainty
    P_tau_0: Prior probability a given system is a lens
    cosmo_type: Which cosmology formulation to infer
    photometric: Whether to assume photometric redshifts (True) or spectroscopic redshifts (False)
    contaminated: Whether to assume the sample is contaminated with non-lenses (True) not (False)
    H0: Hubble constant (in km/s/Mpc)
    key: Random seed for JAX
    early_return: Used for complilation purposes.
    wa_const: Whether to assume wa is constant (True) or not (False)
    w0_const: Whether to assume w0 is constant (True) or not (False)
    use_true_z_phot_code: Whether to use the true zL and zS values in the likelihood (True) or not (False), rather than using the observed values (i.e. including measurement error)
    zL_true: True lens redshift
    zS_true: True source redshift
    fixed_alpha: Whether to use fixed alpha hyperparameters describing the r_obs distribution of false-positives (True) or not (False)
    alpha_dict: Dictionary of alpha hyperparameters (if fixed_alpha=True)
    fixed_beta_gamma: Whether to use fixed beta and gamma hyperparameters describing the lens and source redshift distributions of false-positives (True) or not (False)
    beta_dict: Dictionary of beta hyperparameters (if fixed_beta_gamma=True)
    gamma_dict: Dictionary of gamma hyperparameters (if fixed_beta_gamma=True)
    beta_gamma_lens: Whether to include hyperparameters describing the lens and source redshift distributions of true lenses (True) or not (False)
    fixed_beta_gamma_lens: Whether to use fixed beta and gamma hyperparameters describing the lens and source redshift distributions of true lenses (True) or not (False)
    beta_lens_dict: Dictionary of beta hyperparameters (if fixed_beta_gamma_lens=True)
    gamma_lens_dict: Dictionary of gamma hyperparameters (if fixed_beta_gamma_lens=True)
    '''
    s8 = 0.8;n_s = 0.96;Ob=0; #Adding fiducial cosmological parameters.
    # The priors on lens and source redshifts, as well as any population redshift hyperparameters for photometric systems are added below:
    if photometric:
        print('Assuming photometric redshifts')
        zL_sigma = sigma_zL_obs;zS_sigma = sigma_zS_obs
        if use_true_z_phot_code:
            zL = zL_true
            zS = zS_true
        # Assumes a lognormal relation for P(zS|zL), with a linear dependence of the lognormal hyperparameters on redshift:
        if fixed_s:
            print('Using fixed P(zL|zS) dependence in likelihood')
            s_m = numpyro.deterministic('s_m',s_dict['s_m'])
            s_c = numpyro.deterministic('s_c',s_dict['s_c'])
            scale_m = numpyro.deterministic('scale_m',s_dict['scale_m'])
            scale_c = numpyro.deterministic('scale_c',s_dict['scale_c'])
        else:
            s_m = jnp.squeeze(numpyro.sample("s_m", dist.Uniform(-1,0),sample_shape=(1,),rng_key=key)) #-0.2
            s_c = jnp.squeeze(numpyro.sample("s_c", dist.Uniform(0.01,2),sample_shape=(1,),rng_key=key)) #0.6
            scale_m = jnp.squeeze(numpyro.sample("scale_m", dist.Uniform(0,6),sample_shape=(1,),rng_key=key)) #1.0
            scale_c =  jnp.squeeze(numpyro.sample("scale_c", dist.Uniform(0.1,5),sample_shape=(1,),rng_key=key)) #1.0
        s_z_min = 0.05
        s_z = jnp.array([[s_c+s_m*adjusted_zL_array[elem_zL][elem_zS] for elem_zS in range(len(zS_array))]\
                                                                        for elem_zL in range(len(zL_array))])
        sc_z = jnp.array([[scale_c+scale_m*adjusted_zL_array[elem_zL][elem_zS] for elem_zS in range(len(zS_array))]\
                                                                                for elem_zL in range(len(zL_array))])
        s_z = jnp.where(s_z<s_z_min,s_z_min,s_z)
    else:
        # If photometric=False, the redshift measurements are assumed to be perfect, with zero uncertainty.
        print('Assuming spectroscopic redshifts')
        if use_true_z_phot_code:
            zL = zL_true
            zS = zS_true
        else: 
            zL = zL_obs
            zS = zS_obs
    # Prior on Omega_M:
    OM = jnp.squeeze(numpyro.sample("OM", dist.Uniform(0,1),sample_shape=(1,),rng_key=key))
    if cosmo_type in ['FlatLambdaCDM','FlatwCDM']:
        print('Assuming a flat universe')
        Ok = numpyro.deterministic('Ok',0.0)
        Ode = numpyro.deterministic('Ode',1-(OM+Ok))
    else:
        print('Assuming the universe may have curvature')
        # Prior on Omega_lambda:
        Ode = jnp.squeeze(numpyro.sample("Ode", dist.Uniform(0,1),sample_shape=(1,),rng_key=key))
        Ok = numpyro.deterministic('Ok',1-(OM+Ode))
    if cosmo_type in ['LambdaCDM','FlatLambdaCDM']:
        print('Assuming universe has a cosmological constant')
        # Prior on w0:
        w = numpyro.deterministic('w',-1.0)
        # Prior on wa:
        wa = numpyro.deterministic('wa',0.0)
    elif wa_const == True and w0_const == False:
        print('Assuming a non-evolving dark energy equation of state')
        w = jnp.squeeze(numpyro.sample("w", dist.Uniform(-3,1),sample_shape=(1,),rng_key=key))
        wa = numpyro.deterministic('wa',0.0)
    elif wa_const == False and w0_const == True:
        print('Assuming an evolving dark energy equation of state, but with w0 fixed at -1.')
        w = numpyro.deterministic('w',-1.0)
        wa = jnp.squeeze(numpyro.sample("wa", dist.Uniform(-3,1),sample_shape=(1,),rng_key=key))
    else:
        print('Assuming non-trivial dark energy equation of state')
        w = jnp.squeeze(numpyro.sample("w", dist.Uniform(-3,1),sample_shape=(1,),rng_key=key))
        wa = jnp.squeeze(numpyro.sample("wa", dist.Uniform(-2,2),sample_shape=(1,),rng_key=key))
    cosmo = jc.Cosmology(Omega_c=OM, h=H0/100, Omega_k=Ok, w0=w,
                         Omega_b=Ob, wa=wa, sigma8=s8, n_s=n_s)
    # Calculating the theoretical 'r' ratio for given lens/source redshifts and cosmology:
    if photometric: 
        if cosmo_type in ['FlatLambdaCDM','FlatwCDM']: 
            r_theory = jnp.array([[j_r_SL_flat(adjusted_zL_array[elem_zL][elem_zS],
                                               adjusted_zS_array[elem_zL][elem_zS],cosmo) for elem_zS in range(len(zS_array))] for elem_zL in range(len(zL_array))])
        else: 
            r_theory = jnp.array([[j_r_SL(adjusted_zL_array[elem_zL][elem_zS],
                                          adjusted_zS_array[elem_zL][elem_zS],cosmo) for elem_zS in range(len(zS_array))] for elem_zL in range(len(zL_array))])
    else: 
        if cosmo_type in ['FlatLambdaCDM','FlatwCDM']: r_theory = jnp.array(j_r_SL_flat(zL,zS,cosmo))
        else: r_theory = jnp.array(j_r_SL(zL,zS,cosmo))
    if early_return: return
    if contaminated:
        P_tau = P_tau_0
        P_tau = P_tau.astype('float') #Needs to be a float for dist.Categorical to work
    if photometric and contaminated:
        print('Assuming contaminated, with photometric redshifts')     
        if fixed_alpha:
            N_comp = len(alpha_dict['mu'])
            alpha_mu_dict = {elem:numpyro.deterministic(f'alpha_mu_{elem}',alpha_dict['mu'][elem]) for elem in range(N_comp)}
            alpha_scale_dict = {elem:numpyro.deterministic(f'alpha_scale_{elem}',alpha_dict['scale'][elem]) for elem in range(N_comp)}
            simplex_sample = numpyro.deterministic('alpha_weights',alpha_dict['weights'])
        else:
            N_comp = 4
            alpha_mu_dict = {elem:numpyro.sample(f'alpha_mu_{elem}',dist.Uniform(0,6),sample_shape=(1,)) for elem in range(N_comp)}
            alpha_scale_dict = {elem:numpyro.sample(f'alpha_scale_{elem}',dist.LogUniform(0.001,10),sample_shape=(1,)) for elem in range(N_comp)}
            simplex_sample = numpyro.sample('alpha_weights',dist.Dirichlet(concentration=jnp.array([1.0]*N_comp)))
        if beta_gamma_lens:
            if fixed_beta_gamma_lens:
                beta_lens_mu = {elem:numpyro.deterministic(f'betaLens_mu_{elem}',beta_lens_dict['mu'][elem]) for elem in range(len(beta_lens_dict['mu']))}
                beta_lens_scale = {elem:numpyro.deterministic(f'betaLens_scale_{elem}',beta_lens_dict['scale'][elem]) for elem in range(len(beta_lens_dict['scale']))}
                beta_lens_weights = numpyro.deterministic('betaLens_weights',beta_lens_dict['weights'])
                if not remove_gamma_lens:
                    gamma_lens_mu = {elem:numpyro.deterministic(f'gammaLens_mu_{elem}',gamma_lens_dict['mu'][elem]) for elem in range(len(gamma_lens_dict['mu']))}
                    gamma_lens_scale = {elem:numpyro.deterministic(f'gammaLens_scale_{elem}',gamma_lens_dict['scale'][elem]) for elem in range(len(gamma_lens_dict['scale']))}
                    gamma_lens_weights = numpyro.deterministic('gammaLens_weights',gamma_lens_dict['weights'])
            else:
                beta_lens_mu = {elem:numpyro.sample(f'betaLens_mu_{elem}',dist.Uniform(0,2),sample_shape=(1,),rng_key=key) for elem in range(N_comp)}
                beta_lens_scale = {elem:numpyro.sample(f'betaLens_scale_{elem}',dist.LogUniform(0.001,1),sample_shape=(1,),rng_key=key) for elem in range(N_comp)}
                beta_lens_weights = numpyro.sample('betaLens_weights',dist.Dirichlet(concentration=jnp.array([1.0]*N_comp)),rng_key=key)
                #
                #Gamma Lens Prior:
                if not remove_gamma_lens:
                    gamma_lens_mu = {elem:numpyro.sample(f'gammaLens_mu_{elem}',dist.Uniform(0,10),sample_shape=(1,),rng_key=key) for elem in range(N_comp)}
                    gamma_lens_scale = {elem:numpyro.sample(f'gammaLens_scale_{elem}',dist.LogUniform(0.001,2),sample_shape=(1,),rng_key=key) for elem in range(N_comp)}
                    gamma_lens_weights = numpyro.sample('gammaLens_weights',dist.Dirichlet(concentration=jnp.array([1.0]*N_comp)),rng_key=key)
        if fixed_beta_gamma:
            beta_mu = {elem:numpyro.deterministic(f'beta_mu_{elem}',beta_dict['mu'][elem]) for elem in range(len(beta_dict['mu']))}
            beta_scale = {elem:numpyro.deterministic(f'beta_scale_{elem}',beta_dict['scale'][elem]) for elem in range(len(beta_dict['scale']))}
            beta_weights = numpyro.deterministic('beta_weights',beta_dict['weights'])
            gamma_mu = {elem:numpyro.deterministic(f'gamma_mu_{elem}',gamma_dict['mu'][elem]) for elem in range(len(gamma_dict['mu']))}
            gamma_scale = {elem:numpyro.deterministic(f'gamma_scale_{elem}',gamma_dict['scale'][elem]) for elem in range(len(gamma_dict['scale']))}
            gamma_weights = numpyro.deterministic('gamma_weights',gamma_dict['weights'])
        else:
            beta_mu = {elem:numpyro.sample(f'beta_mu_{elem}',dist.Uniform(0,2),sample_shape=(1,),rng_key=key) for elem in range(N_comp)}
            beta_scale = {elem:numpyro.sample(f'beta_scale_{elem}',dist.LogUniform(0.001,1),sample_shape=(1,),rng_key=key) for elem in range(N_comp)}
            beta_weights = numpyro.sample('beta_weights',dist.Dirichlet(concentration=jnp.array([1.0]*N_comp)),rng_key=key)
            #
            #Gamma Prior:
            gamma_mu = {elem:numpyro.sample(f'gamma_mu_{elem}',dist.Uniform(0,10),sample_shape=(1,),rng_key=key) for elem in range(N_comp)}
            gamma_scale = {elem:numpyro.sample(f'gamma_scale_{elem}',dist.LogUniform(0.001,2),sample_shape=(1,),rng_key=key) for elem in range(N_comp)}
            gamma_weights = numpyro.sample('gamma_weights',dist.Dirichlet(concentration=jnp.array([1.0]*N_comp)),rng_key=key)
        def photometric_and_contaminated_likelihood(r_obs,sigma_r_obs,r_theory,
                                            zL_obs,zS_obs,
                                            zL_sigma,zS_sigma,
                                            zL_array,zS_array,
                                            P_tau=None,
                                            prob_2=None,prob_3 = None,
                                            prob_zL_zS = 1.0,z_diff = None):
            prob_r_obs_TP = jax_truncnorm.logpdf(x=r_obs/sigma_r_obs,
                                            loc=r_theory/sigma_r_obs, 
                                            scale=1,
                                            a=-r_theory/sigma_r_obs,b=np.inf) - jnp.log(sigma_r_obs)
            #Downweighting the systems for which zL_true>zS_true, as this isn't physical:
            prob_r_obs_TP = jnp.where(redshift_bool,prob_r_obs_TP,-500+prob_r_obs_TP)
            prob_r_obs_FP = truncnorm_mixture(weights = jnp.array([simplex_sample.T[elem] for elem in range(4)]),
                                        loc = jnp.array([alpha_mu_dict[elem] for elem in range(4)]),
                                        scale= jnp.array([alpha_scale_dict[elem] for elem in range(4)]),
                                        obs = r_obs)
            prob_zL_FP = truncnorm_mixture(
                                    weights = jnp.array([beta_weights.T[elem] for elem in range(len(beta_weights))]),
                                    loc = jnp.array([beta_mu[elem] for elem in range(len(beta_mu.keys()))]),
                                    scale= jnp.array([beta_scale[elem] for elem in range(len(beta_scale.keys()))]),
                                    obs = zL_obs)
            prob_zS_FP = truncnorm_mixture(
                                    weights = jnp.array([gamma_weights.T[elem] for elem in range(len(gamma_weights))]),
                                    loc = jnp.array([gamma_mu[elem] for elem in range(len(gamma_mu.keys()))]),
                                    scale= jnp.array([gamma_scale[elem] for elem in range(len(gamma_scale.keys()))]),
                                    obs = zS_obs)
            if beta_gamma_lens:
                prob_zL_TP = jnp.array([[truncnorm_mixture(
                                            weights = jnp.array([beta_lens_weights.T[elem] for elem in range(len(beta_lens_weights))]),
                                            loc = jnp.array([beta_lens_mu[elem] for elem in range(len(beta_lens_mu.keys()))]),
                                            scale= jnp.array([beta_lens_scale[elem] for elem in range(len(beta_lens_scale.keys()))]),
                                            obs = zL_array[elem_zL]) for elem_zS in range(len(zS_array))] for elem_zL in range(len(zL_array))])
                if remove_gamma_lens:
                    prob_zS_TP = 0.0
                else:
                    prob_zS_TP = jnp.array([[truncnorm_mixture(
                                            weights = jnp.array([gamma_lens_weights.T[elem] for elem in range(len(gamma_lens_weights))]),
                                            loc = jnp.array([gamma_lens_mu[elem] for elem in range(len(gamma_lens_mu.keys()))]),
                                            scale= jnp.array([gamma_lens_scale[elem] for elem in range(len(gamma_lens_scale.keys()))]),
                                            obs = zS_array[elem_zS]) for elem_zS in range(len(zS_array))] for elem_zL in range(len(zL_array))])
            else:
                prob_zL_TP = 0.0
                prob_zS_TP = 0.0
            #'log_prob' finds the natural logarithm (not log10), hence these are natural-logged:
            prob_zL_zS = jax_lognormal(x=z_diff,
                            s=s_z,
                            loc=0, #Defined to be 0 
                            scale=sc_z).pdf()
            prob =  likelihood_phot_contam(P_tau,prob_r_obs_TP,prob_r_obs_FP,prob_2,prob_3,prob_zL_zS,
                                                prob_zL_FP,prob_zS_FP,
                                                prob_zL_TP,prob_zS_TP,
                                                delta_zL,delta_zS,
                                                trapezium_factor_zL,trapezium_factor_zS,
                                                redshift_bool,
                                                trapezium_factor,
                                                ) #prob_zL_zS is not logged
            prob = jnp.where(Ode*jnp.ones(len(prob))<0,-np.inf,prob)           
            prob = jnp.where(Ode*jnp.ones(len(prob))>1,-np.inf,prob)
            return prob                                             
        prob = photometric_and_contaminated_likelihood(
                                            r_obs,sigma_r_obs,r_theory,
                                            zL_obs,zS_obs,
                                            zL_sigma,zS_sigma,
                                            zL_array,zS_array,
                                            P_tau=P_tau,
                                            prob_2 = prob_2,prob_3 = prob_3,
                                            prob_zL_zS = prob_zL_zS,
                                            z_diff = z_diff)
        L = numpyro.factor("Likelihood",prob)
    else:
        print('Assuming not contaminated, with spectroscopic redshifts')
        assert not photometric and not contaminated
        prob = dist.TruncatedNormal(r_theory, sigma_r_obs, low = 0).log_prob(r_obs)
        prob = jnp.where(Ode*jnp.ones(len(prob))<0,-np.inf,prob)
        prob = jnp.where(Ode*jnp.ones(len(prob))>1,-np.inf,prob)
        prob = jnp.where(Ode<0,-np.inf,prob)
        prob = jnp.where(Ode>1,-np.inf,prob) 
        numpyro.factor('Likelihood',prob)    
 
# Code to run the cosmology posterior sampling:
def run_MCMC(photometric,contaminated,cosmo_type,
            zL_obs,zS_obs,sigma_zL_obs,sigma_zS_obs,
            r_obs,sigma_r_obs,P_tau_0,
            num_warmup = 200,num_samples=1000,num_chains=2,
            H0=np.nan,target_accept_prob=0.8,
            wa_const=False,w0_const=False,key_int = 0,
            zL_true=None,zS_true=None,
            r_true = None,
            use_true_z_phot_code=False,
            fixed_alpha=False,alpha_dict = {},
            fixed_beta_gamma=False,beta_dict = {},gamma_dict = {},
            beta_gamma_lens=False,fixed_beta_gamma_lens = False,beta_lens_dict = {},gamma_lens_dict = {},
            remove_gamma_lens=False,s_dict = {},fixed_s = False,):
    print('Random key:',key_int)
    zL_array,zS_array,delta_zL,delta_zS,trapezium_factor_zL,trapezium_factor_zS,redshift_bool,\
            adjusted_zL_array,adjusted_zS_array,trapezium_factor,z_diff = calculate_integral_variables(zL_obs,zS_obs,sigma_zL_obs,sigma_zS_obs)
    prob_2 = jnp.array([[jax_truncnorm.logpdf(x=zL_obs/sigma_zL_obs,loc=zL_array[elem_zL]/sigma_zL_obs, scale=1,
                                                         a=-zL_array[elem_zL]/sigma_zL_obs,b=np.inf) - jnp.log(sigma_zL_obs) \
                                                         for elem_zS in range(len(zS_array))]\
                                                         for elem_zL in range(len(zL_array))])
    prob_3 = jnp.array([[jax_truncnorm.logpdf(x=zS_obs/sigma_zS_obs,loc=zS_array[elem_zS]/sigma_zS_obs, scale=1,
                                                         a=-zS_array[elem_zS]/sigma_zS_obs,b=np.inf) - jnp.log(sigma_zS_obs) \
                                                         for elem_zS in range(len(zS_array))] \
                                                         for elem_zL in range(len(zL_array))])
    model_args = {'zL_obs':zL_obs,'zS_obs':zS_obs,
                'sigma_zL_obs':sigma_zL_obs,'sigma_zS_obs':sigma_zS_obs,
                'r_obs':r_obs,'sigma_r_obs':sigma_r_obs,
                'P_tau_0':P_tau_0,'cosmo_type':cosmo_type,
                'photometric':photometric,'contaminated':contaminated,
                'H0':H0,
                'wa_const':wa_const,'w0_const':w0_const,
                'use_true_z_phot_code':use_true_z_phot_code,'zL_true':zL_true,'zS_true':zS_true,
                'fixed_alpha':fixed_alpha,'alpha_dict':alpha_dict,
                'fixed_beta_gamma':fixed_beta_gamma,
                'beta_dict':beta_dict,'gamma_dict':gamma_dict,
                'beta_gamma_lens':beta_gamma_lens,'fixed_beta_gamma_lens':fixed_beta_gamma_lens,
                'beta_lens_dict':beta_lens_dict,'gamma_lens_dict':gamma_lens_dict,
                'zL_array':zL_array,'zS_array':zS_array,
                'delta_zL':delta_zL,'delta_zS':delta_zS,
                'trapezium_factor_zL':trapezium_factor_zL,'trapezium_factor_zS':trapezium_factor_zS,
                'redshift_bool':redshift_bool,
                'adjusted_zL_array':adjusted_zL_array,'adjusted_zS_array':adjusted_zS_array,
                'prob_2':prob_2,'prob_3':prob_3,'trapezium_factor':trapezium_factor,
                'remove_gamma_lens':remove_gamma_lens,'z_diff':z_diff,
                's_dict':s_dict,'fixed_s':fixed_s}
    key = jax.random.PRNGKey(key_int)
    st = time.time()
    j_likelihood_SL(**model_args,key=key,early_return=True)
    mt=time.time()
    j_likelihood_SL(**model_args,key=key,early_return=True)
    et=time.time()
    print('Uncompiled time',mt-st)
    print('Compiled time',et-mt)
    outer_kernel =  NUTS(model = j_likelihood_SL,target_accept_prob = target_accept_prob,init_strategy=init_to_uniform)
    sampler_0 = MCMC(outer_kernel,
                    num_warmup=num_warmup,
                    num_samples=num_samples,
                    num_chains=num_chains,
                    progress_bar=True)
    print('Starting Warmup:')
    sampler_0.warmup(key,**model_args,collect_warmup=True)
    ##
    print("Starting main run:")
    sampler_0.run(key,**model_args,key=None)
    return sampler_0
