'''
Code description: This file infers the cosmological posteriors (as well as those for various population hyperparameters) given a sample of impure and/or photometric
strong lenses.
The MCMC is run using NUTS (https://num.pyro.ai/en/latest/mcmc.html#numpyro.infer.hmc.NUTS).
The code allows for use of HMCECS (i.e. batching), though this can introduce unexpected biases so should be avoided wherever possible (and will likely be removed
in subsequent commits).
'''

from astropy.cosmology import LambdaCDM,FlatLambdaCDM,wCDM,FlatwCDM,w0waCDM
from JAX_samples_to_dict import JAX_samples_to_dict
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
import gc

class truncnorm_class:
    def __init__(self,loc,scale,a,b):
        self.loc = loc
        self.scale = scale
        self.a = a
        self.b = b
    def logpdf(self,x):
        return jax_truncnorm.logpdf(x=x,a=self.a,b=self.b,loc=self.loc,scale=self.scale)

@jit
def MVN_samp(loc_0,loc_1,sig_0,sig_1,x0,x1,sigma_01,sigma_10):
        return dist.MultivariateNormal(loc=jnp.array([loc_0,loc_1]),
                        covariance_matrix=jnp.array([[sig_0**2,sigma_01**2],
                                                     [sigma_10**2,sig_1**2]])).log_prob(jnp.array([x0,x1]).T) 

@jit
def likelihood_PC(P_tau,prob_1a,prob_1b,prob_2,prob_3,prob_4a,prob_4b,prob_5a,prob_5b):
        return (jnp.log(P_tau*jnp.exp(prob_1a)*jnp.exp(prob_4a)*jnp.exp(prob_5a)+\
                       (1-P_tau)*jnp.exp(prob_1b)*jnp.exp(prob_4b)*jnp.exp(prob_5b))+prob_2+prob_3)

@jit
def likelihood_PC_no_parent(P_tau,prob_1a,prob_1b,prob_2,prob_3,prob_zL_zS): #prob_zL_zS is not logged
        # jax.debug.print('Likelihood_pc Output: {a},{b},{c},{d},{e},{f},{g}',a=P_tau,b=jnp.exp(prob_1a),c=(1-P_tau),d=jnp.exp(prob_1b),e=prob_2,f=prob_3,g=prob_zL_zS)
        # jax.debug.print('To be logged {a}',a=jnp.min(P_tau*jnp.exp(prob_1a)+(1-P_tau)*jnp.exp(prob_1b)))
        likelihood_p1 = P_tau*jnp.exp(prob_1a)*prob_zL_zS
        likelihood_p2 = (1-P_tau)*jnp.exp(prob_1b)
        likelihood_prelim = jnp.log(likelihood_p1+likelihood_p2)
        #Accounting for rounding errors - when p1 is zero, can just use the logged version of p2 alone, and vice versa.
        likelihood_prelim = jnp.where(likelihood_p1==0,jnp.log(1-P_tau)+prob_1b,likelihood_prelim)
        likelihood_prelim = jnp.where(likelihood_p2==0,jnp.log(P_tau)+prob_1a+jnp.log(prob_zL_zS),likelihood_prelim)
        return likelihood_prelim+prob_2+prob_3

def breakpoint_if_nonfinite(prob,zL,zS,r_theory,OM,Ode,Ok,w,wa):
  is_finite = jnp.isfinite(prob).all()
  def true_fn(prob,zL,zS,r_theory,OM,Ode,Ok,w,wa):
    pass
  def false_fn(prob,zL,zS,r_theory,OM,Ode,Ok,w,wa):
    jax.debug.breakpoint()
  jax.lax.cond(is_finite, true_fn, false_fn, prob,zL,zS,r_theory,OM,Ode,Ok,w,wa)

def breakpoint_if_nonfinite_0(prob,alpha_mu,alpha_scale,alpha_s,r_theory,r_theory_2):
  is_finite = jnp.isfinite(prob).all()
  def true_fn(prob,alpha_mu,alpha_scale,alpha_s,r_theory,r_theory_2):
    pass
  def false_fn(prob,alpha_mu,alpha_scale,alpha_s,r_theory,r_theory_2):
    jax.debug.breakpoint()
  jax.lax.cond(is_finite, true_fn, false_fn, prob,alpha_mu,alpha_scale,alpha_s,r_theory,r_theory_2)

def j_likelihood_SL_batch(zL_obs,zS_obs,sigma_zL_obs,sigma_zS_obs,r_obs,sigma_r_obs,sigma_r_obs_2=[np.nan],P_tau_0 = [],cosmo_type='',
                    photometric=False,contaminated=False,H0=np.nan,key=None,
                    likelihood_check=False,likelihood_dict = {},cov_redshift=False,early_return=False,
                    batch_bool=True, wa_const = False, w0_const = False,GMM_zL = False,GMM_zS = False,
                    fixed_GMM = False, GMM_zL_dict = {}, GMM_zS_dict = {},spec_indx = [],no_parent=False,
                    trunc_zL=False,trunc_zS=False,P_tau_dist = False,sigma_P_tau = [],lognorm_parent = False,
                    unimodal_beta=True,bimodal_beta=False,true_zL_zS_dep = False,
                    prior_dict = {},batch_number=np.nan
                    ):
    #Other permutations haven't yet been tested:
    assert photometric and contaminated and lognorm_parent
    assert GMM_zL==False and GMM_zS==False
    assert not fixed_GMM and not cov_redshift
    OM = prior_dict['OM'];Ok = prior_dict['Ok'];Ode = prior_dict['Ode'];w = prior_dict['w'];wa = prior_dict['wa']
    s_m = prior_dict['s_m'];s_c = prior_dict['s_c'];scale_m = prior_dict['scale_m'];scale_c = prior_dict['scale_c']
    alpha_mu_dict = prior_dict['alpha_mu_dict'];alpha_scale_dict = prior_dict['alpha_scale_dict']
    simplex_sample = prior_dict['simplex_sample']
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
    P_tau_dist: Whether to use a distribution for P_tau, rather than a single value
    sigma_P_tau: Width of P_tau distribution, where applicable.
    lognorm_parent: Boolean for whether to include zL vs zS dependence (i.e. P(zS|zL)).
    '''
    if lognorm_parent: assert not GMM_zL and not GMM_zS; assert no_parent
    if P_tau_dist: assert isinstance(sigma_P_tau,float) or len(sigma_P_tau)==len(P_tau_0)
    sigma_r_obs_2 = 100# sigma_r_obs
    # Lens and source redshift parent hyperparameters
    mu_zL_g_L=None;mu_zS_g_L = None # For true lenses 
    mu_zL_g_NL=None;mu_zS_g_NL = None # For non-lenses
    sigma_zL_g_L=None;sigma_zS_g_L=None
    sigma_zL_g_NL=None;sigma_zS_g_NL=None
    # Lens and source redshift parent hyperparameters (assuming gaussian mixture model):
    w_zL=None;w_zS=None
    mu_zL_g_L_A=None;mu_zL_g_L_B = None
    sigma_zL_g_L_A=None;sigma_zL_g_L_B=None
    sigma_01_g_L=None;sigma_01_g_NL=None
    sigma_10_g_L=None;sigma_10_g_NL=None
    if likelihood_check: #For bug-checking purposes
        phot_indx = list(set(np.arange(len(likelihood_dict['zL_obs'])).tolist())-set(spec_indx))
        if len(likelihood_dict['zL_obs'])<8000: subsample_size = len(likelihood_dict['zL_obs'])//2
        if len(likelihood_dict['zL_obs'])<=12000: subsample_size = 8000
        else: subsample_size = 12000
    if not likelihood_check:
        phot_indx = list(set(np.arange(len(zL_obs)).tolist())-set(spec_indx))
        if len(zL_obs)<8000: subsample_size=len(zL_obs)//2
        elif len(zL_obs)<=12000: subsample_size = 8000
        else: subsample_size = 12000
    spec_indx = jnp.array(spec_indx).astype('int');phot_indx = jnp.array(phot_indx).astype('int') 
    #assert False #NEED TO ADD IN TRUNCATION TO THE GMM PARENT MODEL!
    #assert False #Also the MCMC redshift-only inference should definitely match the best fit value right??
    s8 = 0.8;n_s = 0.96;Ob=0; #Putting all the matter in dark-matter (doesn't make a difference)
    if not likelihood_check:
        # The priors on lens and source redshifts, as well as any population redshift hyperparameters for photometric systems are added below:
        if photometric:
            print('Assuming photometric redshifts')
            zL_sigma = sigma_zL_obs;zS_sigma = sigma_zS_obs
            # Removing assertion that zS has to be > zL - still bugs:
            # assert False #Think these should all be uniform distributions (at least zL, zS), as marginalising over them here???? Then the normal distribution appears in the likelihood term.
            # zL = numpyro.sample('zL',dist.TruncatedNormal(jnp.array(zL_obs),zL_sigma,low=0),sample_shape=(1,),rng_key=key).flatten()
            # zS = numpyro.sample('zS',dist.TruncatedNormal(jnp.array(zS_obs),zS_sigma,low=0),sample_shape=(1,),rng_key=key).flatten()
            #
            zL_obs_low_lim = jnp.array(zL_obs-4*zL_sigma)
            zL_obs_low_lim = zL_obs_low_lim*(zL_obs_low_lim>0) #Minimum value is 0
            zL_obs_up_lim = jnp.array(zL_obs+4*zL_sigma)
            zL = numpyro.sample(f'zL_B{batch_number}',dist.Uniform(low = zL_obs_low_lim,high = zL_obs_up_lim),sample_shape=(1,),rng_key=key).flatten()
            ##Minimum value is zL. First term: zS if zS>0, else 0. Second term, zL if zS<zL, else 0
            zS_obs_low_lim = jnp.array(zS_obs-5*zS_sigma)
            zS_obs_low_lim = zS_obs_low_lim*(zS_obs_low_lim>zL) + zL*(zS_obs_low_lim<zL)
            zS_obs_low_lim = zS_obs_low_lim*(zS_obs_low_lim>0)
            zS_obs_up_lim = jnp.array(zS_obs+5*zS_sigma)
            zS = numpyro.sample(f'zS_B{batch_number}',dist.Uniform(low = zS_obs_low_lim, high = zS_obs_up_lim),sample_shape=(1,),rng_key=key).flatten()
            #
            # Assumes a lognormal relation for P(zS|zL), with a linear dependence of the lognormal hyperparameters on redshift:
            if not no_parent:
                if GMM_zL:
                    if fixed_GMM:
                        print('Using fixed parent lens redshift distribution')
                        mu_zL_g_L_A = numpyro.deterministic('mu_zL_g_L_A',GMM_zL_dict['mu_zL_g_L_A'])
                        sigma_zL_g_L_A = numpyro.deterministic('sigma_zL_g_L_A',GMM_zL_dict['sigma_zL_g_L_A'])
                        mu_zL_g_L_B = numpyro.deterministic('mu_zL_g_L_B',GMM_zL_dict['mu_zL_g_L_B'])
                        sigma_zL_g_L_B = numpyro.deterministic('sigma_zL_g_L_B',GMM_zL_dict['sigma_zL_g_L_B'])
                        w_zL = numpyro.deterministic('w_zL',GMM_zL_dict['w_zL'])
                    else:
                        mu_zL_g_L_A = jnp.squeeze(numpyro.sample("mu_zL_g_L_A", dist.Uniform(0,4),sample_shape=(1,),rng_key=key))
                        sigma_zL_g_L_A = jnp.squeeze(numpyro.sample("sigma_zL_g_L_A", dist.Uniform(0.01,5),sample_shape=(1,),rng_key=key))
                        mu_zL_g_L_B = jnp.squeeze(numpyro.sample("mu_zL_g_L_B", dist.Uniform(0,4),sample_shape=(1,),rng_key=key))
                        sigma_zL_g_L_B = jnp.squeeze(numpyro.sample("sigma_zL_g_L_B", dist.Uniform(0.01,5),sample_shape=(1,),rng_key=key))
                        w_zL = numpyro.sample('w_zL', dist.Uniform(0.5,1),rng_key=key) #Component 1 must be the largest component
                else:
                    mu_zL_g_L = jnp.squeeze(numpyro.sample("mu_zL_g_L", dist.Uniform(0.1,1.5),sample_shape=(1,),rng_key=key))
                    sigma_zL_g_L = jnp.squeeze(numpyro.sample("sigma_zL_g_L", dist.Uniform(0.01,1),sample_shape=(1,),rng_key=key))
                if GMM_zS:
                    if fixed_GMM:
                        print('Using fixed parent source redshift distribution')
                        mu_zS_g_L_A = numpyro.deterministic('mu_zS_g_L_A',GMM_zS_dict['mu_zS_g_L_A'])
                        sigma_zS_g_L_A = numpyro.deterministic('sigma_zS_g_L_A',GMM_zS_dict['sigma_zS_g_L_A'])
                        mu_zS_g_L_B = numpyro.deterministic('mu_zS_g_L_B',GMM_zS_dict['mu_zS_g_L_B'])
                        sigma_zS_g_L_B = numpyro.deterministic('sigma_zS_g_L_B',GMM_zS_dict['sigma_zS_g_L_B'])
                        w_zS = numpyro.deterministic('w_zS',GMM_zS_dict['w_zS'])
                    else:
                        mu_zS_g_L_A = jnp.squeeze(numpyro.sample("mu_zS_g_L_A", dist.Uniform(0,4),sample_shape=(1,),rng_key=key))
                        sigma_zS_g_L_A = jnp.squeeze(numpyro.sample("sigma_zS_g_L_A", dist.Uniform(0.01,5),sample_shape=(1,),rng_key=key))
                        mu_zS_g_L_B = jnp.squeeze(numpyro.sample("mu_zS_g_L_B", dist.Uniform(0,4),sample_shape=(1,),rng_key=key))
                        sigma_zS_g_L_B = jnp.squeeze(numpyro.sample("sigma_zS_g_L_B", dist.Uniform(0.01,5),sample_shape=(1,),rng_key=key))
                        w_zS = numpyro.sample('w_zS', dist.Uniform(0.5,1),rng_key=key) #Component 1 must be the largest component
                else:
                    mu_zS_g_L = jnp.squeeze(numpyro.sample("mu_zS_g_L", dist.Uniform(0.1,2),sample_shape=(1,),rng_key=key))
                    sigma_zS_g_L = jnp.squeeze(numpyro.sample("sigma_zS_g_L", dist.Uniform(0.01,1.5),sample_shape=(1,),rng_key=key))
            # Adding a covariance term between zL and zS (this is no longer in use):
            if cov_redshift:
                assert not GMM_zL and not GMM_zS
                sigma_01_g_L =  jnp.squeeze(numpyro.sample("sigma_01_g_L", dist.Uniform(0.01,2),sample_shape=(1,),rng_key=key))
                sigma_10_g_L =  jnp.squeeze(numpyro.sample("sigma_10_g_L", dist.Uniform(0.01,2),sample_shape=(1,),rng_key=key))
            else:
                sigma_01_g_L =  0.0;sigma_10_g_L =  0.0
        else:
            # If photometric=False, the redshift measurements are assumed to be perfect, with zero uncertainty.
            print('Assuming spectroscopic redshifts')
            zL = zL_obs #Think still need to have an error-budget when using spectroscopic redshifts?
            zS = zS_obs
        if likelihood_check: # For bug-checking purposes:
            OM = likelihood_dict['OM'];Ok = likelihood_dict['Ok']
            Ode = numpyro.deterministic('Ode',1-(OM+Ok))
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
            try:
                P_tau_0 = likelihood_dict['P_tau'];sigma_r_obs_2 = likelihood_dict['sigma_r_obs_2']
            except: pass
    cosmo = jc.Cosmology(Omega_c=OM, h=H0/100, Omega_k=Ok, w0=w,
                         Omega_b=Ob, wa=wa, sigma8=s8, n_s=n_s)
    # Calculating the theoretical 'r' ratio for given lens/source redshifts and cosmology:
    if cosmo_type in ['FlatLambdaCDM','FlatwCDM']: r_theory = j_r_SL_flat(zL,zS,cosmo)
    else: r_theory = j_r_SL(zL,zS,cosmo)
    if early_return: return 0
    # If contaminated, need to know the prior probability each system is a lens (P_tau_0):
    if contaminated:
        if P_tau_dist: # Uses a distribution for P_tau, with mean P_tau_0 and width sigma_P_tau, rather than a single value:
            Beta_class_instance = beta_class(mean=P_tau_0,sigma=sigma_P_tau)
            beta_A = Beta_class_instance.A
            beta_B = Beta_class_instance.B
            P_tau = numpyro.sample(f'P_tau_B{batch_number}',dist.Beta(beta_A,beta_B),sample_shape = (1,),rng_key=key).flatten()
        else:
            P_tau = P_tau_0
        P_tau = P_tau.astype('float') #Needs to be a float for dist.Categorical to work
        # r_theory_2 = jnp.squeeze(numpyro.sample("r_theory_2", dist.TruncatedNormal(loc=alpha_mu,scale=alpha_sigma,
        #                                                                             low=0),sample_shape=(1,),rng_key=key))
    if contaminated and not photometric:
        # with numpyro.plate("C_plate",len(P_tau)):
            # Lens_bool = numpyro.sample('Lens_bool',dist.Bernoulli(P_tau),rng_key=key,sample_shape=(1,),infer={'enumerate': 'parallel'}).flatten().astype('float')
            # Lens_bool = numpyro.sample('Lens_bool',dist.Categorical(jnp.array([P_tau, 1-P_tau]).T),sample_shape=(1,),rng_key=key,infer={'enumerate': 'parallel'}).flatten().astype('float')
        print('Assuming contaminated, with spectroscopic redshifts')
        # assert not np.isnan(sigma_r_obs_2)
#         prob_1 = dist.Mixture(dist.Categorical(jnp.array([P_tau, 1-P_tau]).T),
#                             [dist.Normal(r_theory, sigma_r_obs),
#                              dist.Normal(r_theory, sigma_r_obs_2)]).log_prob(r_obs)
#         L = numpyro.factor("Likelihood",prob_1)
        r_theory_2_low_lim = 0 #jnp.array(r_obs-3*sigma_r_obs)
        r_theory_2_up_lim = jnp.max(r_obs+5*sigma_r_obs)*jnp.ones(len(r_obs))
        # r_theory_2_low_lim = r_theory_2_low_lim*(r_theory_2_low_lim>0)
        # r_theory_2_up_lim = jnp.array(r_obs+3*sigma_r_obs)
        r_theory_2 = numpyro.sample('r_theory_2',dist.Uniform(low = r_theory_2_low_lim, high = r_theory_2_up_lim),sample_shape=(1,),rng_key=key).flatten() 
        # 'alpha' refers to the parent hyperparameters to be inferred if a system is not a lens:
        #LogNorm:
        # alpha_mu = numpyro.deterministic('alpha_mu',0) #jnp.squeeze(numpyro.sample("alpha_mu", dist.Uniform(-1,0),sample_shape=(1,),rng_key=key))
        # alpha_scale = jnp.squeeze(numpyro.sample("alpha_scale", dist.Uniform(0.1,5),sample_shape=(1,),rng_key=key))
        # alpha_s = jnp.squeeze(numpyro.sample("alpha_s", dist.Uniform(0.1,5),sample_shape=(1,),rng_key=key))
        #Truncnorm:
        alpha_mu = jnp.squeeze(numpyro.sample("alpha_mu", dist.Uniform(0,2),sample_shape=(1,),rng_key=key))
        alpha_scale = jnp.squeeze(numpyro.sample("alpha_scale", dist.Uniform(0.01,5),sample_shape=(1,),rng_key=key))
        alpha_s = np.nan
        alpha_mu_2 = jnp.squeeze(numpyro.sample("alpha_mu_2", dist.Uniform(0,2),sample_shape=(1,),rng_key=key))
        alpha_scale_2 = jnp.squeeze(numpyro.sample("alpha_scale_2", dist.Uniform(0.01,5),sample_shape=(1,),rng_key=key))
        alpha_w = jnp.squeeze(numpyro.sample("alpha_weights", dist.Uniform(0,1),sample_shape=(1,),rng_key=key))
        def spectroscopic_and_contaminated_likelihood(P_tau,r_obs,r_theory,sigma_r_obs,
                                                      r_theory_2,sigma_r_obs_2,
                                                      alpha_mu,alpha_scale,alpha_s):
            # #Logged in the likelihood:
            # prob_TP1 = jax_truncnorm.pdf(x=r_obs,loc=r_theory,scale=sigma_r_obs,
            #                                     a=-r_theory/sigma_r_obs,b=np.inf)
            # #These SHOULD be pre-logged, otherwise get an additional rounding error, when I multiply them together, then re-log.
            # prob_FP1 = jax_truncnorm.logpdf(x=r_obs,loc=r_theory_2,scale=sigma_r_obs_2,
            #                                     a=-r_theory_2/sigma_r_obs_2,b=np.inf)
            # # prob_FP2 = jax_lognormal(x=r_theory_2,loc=alpha_mu,scale=alpha_scale,s=alpha_s).log_prob()
            # # prob_FP2 = jax_truncnorm.logpdf(x = r_theory_2,loc=alpha_mu,scale=alpha_scale,
            # #                                       a=-alpha_mu/alpha_scale,b=np.inf)
            # prob_FP2 = dist.Normal(loc = alpha_mu,scale = alpha_scale).log_prob(r_theory_2)
            # prob_1 = jnp.log(P_tau*prob_TP1+(1-P_tau)*jnp.exp(prob_FP1+prob_FP2))
            # prob_1 = jnp.log(P_tau)+prob_TP1
            # prob_1 = prob_FP1+prob_FP2
            # prob_1 = jnp.where(r_theory_2<alpha_mu,-np.inf,prob_1)
            #Previously, with a mixture:
            prob_1 = dist.Mixture(dist.Categorical(jnp.array([P_tau, (1-P_tau)*alpha_w,(1-P_tau)*(1-alpha_w)]).T),
                            [dist.TruncatedNormal(r_theory, sigma_r_obs,low=0),
                             dist.TruncatedNormal(alpha_mu, alpha_scale,low=0),
                             dist.TruncatedNormal(alpha_mu_2,alpha_scale_2,low=0)
                             ]).log_prob(r_obs)
            #Previously, without r(alpha) parameterisation:
            # prob_1 = dist.Mixture(dist.Categorical(jnp.array([batch_lens_bool, 1-batch_lens_bool]).T),
            #                 [dist.TruncatedNormal(r_theory, sigma_r_obs,low=0),
            #                 dist.TruncatedNormal(r_theory, sigma_r_obs_2,low=0)]).log_prob(r_obs)
            prob_1 = jnp.where(r_theory_2<0,-np.inf,prob_1)
            prob_1 = jnp.where(Ode*jnp.ones(len(prob_1))<0,-np.inf,prob_1)
            prob_1 = jnp.where(Ode*jnp.ones(len(prob_1))>1,-np.inf,prob_1)
            # jax.debug.print('prob_TP1,prob_FP1,prob_FP2: {a},{b},{c}',a=prob_TP1,b=prob_FP1,c=prob_FP2)
            # jax.debug.print('prob_1 {a}',a=prob_1)
            # jax.debug.print('alpha_s,alpha_mu,alpha_scale: {a},{b},{c}',a=alpha_s,b=alpha_mu,c=alpha_scale)
            # breakpoint_if_nonfinite_0(prob_1,alpha_mu,alpha_scale,alpha_s,r_theory,r_theory_2)
            # jax.debug.print('prob_1 {a}',a=jnp.sum(prob_1))
            return prob_1
        if batch_bool:
            assert False #Note, batching can introduce unexplained errors!!!
            with numpyro.plate("N", r_theory.shape[0], subsample_size=subsample_size):
                # jax.debug.print('OM,Ode,Ok: {a},{b},{c}',a=OM,b=Ode,c=Ok)
                batch_r_obs = numpyro.subsample(r_obs,event_dim=0)
                batch_P_tau =  numpyro.subsample(P_tau,event_dim=0)
                batch_r_theory = numpyro.subsample(r_theory,event_dim=0)
                batch_r_theory_2 = numpyro.subsample(r_theory_2,event_dim=0)
                batch_sigma_r_obs = numpyro.subsample(sigma_r_obs,event_dim=0)
                batch_sigma_r_obs_2 = numpyro.subsample(sigma_r_obs_2,event_dim=0)
                # with numpyro.plate("N2",Lens_bool.shape[-1]):
                # batch_lens_bool = numpyro.subsample(Lens_bool,event_dim=0)
                # NOTE: ValueError: All elements of 'component_distributions' must be instaces of numpyro.distributions.Distribution subclasses.
                # => I.e. can't use a random class or function (e.g. jax_truncnorm) and use it in a mixture.
                #Changing this from a dist.Mixture to use jaxnorm:
                prob_1 = spectroscopic_and_contaminated_likelihood(batch_P_tau,batch_r_obs,r_theory,batch_sigma_r_obs,
                                                                   batch_r_theory_2,batch_sigma_r_obs_2,
                                                                   alpha_mu,alpha_scale,alpha_s)
                L = numpyro.factor("Likelihood",prob_1)
        else:
            prob_1 = spectroscopic_and_contaminated_likelihood(P_tau,r_obs,r_theory,sigma_r_obs,
                                                                r_theory_2,sigma_r_obs_2,
                                                                alpha_mu,alpha_scale,alpha_s)
            L = numpyro.factor("Likelihood",prob_1)
    elif photometric and not contaminated:
        print('Assuming not contaminated, with photometric redshifts')
        def photometric_likelihood(r_obs,sigma_r_obs,r_theory,
                                   r_obs_spec,sigma_r_obs_spec,r_theory_spec,
                                   zL_obs,zS_obs,
                                   zL_sigma,zS_sigma,
                                   zL,zS,
                                   mu_zL_g_L=None,mu_zS_g_L = None,
                                   w_zL=None,w_zS=None,
                                   mu_zL_g_L_A=None,mu_zL_g_L_B = None,
                                   sigma_zL_g_L_A=None,sigma_zL_g_L_B=None,
                                   GMM_zL=False,GMM_zS=False):
            assert len(r_obs_spec)==0 #Otherwise will need to sum up likelihoods as the arrays will be different lengths. 
            #Previously: Not adding in truncation at 0 here, as it introduces bias into the parent redshift distribution:
            #Now: Adding in trucation at 0, as I no longer want to infer the parent redshift distribution, and the distribution of (r_obs-r_true)/sigma_r_obs is significantly non-gaussian.
            # prob_1 = dist.TruncatedNormal(r_theory, sigma_r_obs,low=0).log_prob(r_obs)
            prob_1 = jax_truncnorm.logpdf(x=r_obs,loc=r_theory, scale=sigma_r_obs,a=-r_theory/sigma_r_obs,b=np.inf)
            # prob_1_spec = dist.TruncatedNormal(r_theory_spec, sigma_r_obs_spec,low=0).log_prob(r_obs_spec)
            prob_1_spec = jax_truncnorm.logpdf(x=r_obs_spec,loc=r_theory_spec, scale=sigma_r_obs_spec,a=-r_theory_spec/sigma_r_obs_spec,b=np.inf)
            # prob_1 = jnp.where(r_obs<0,-np.inf,prob_1)
            # prob_1_spec = jnp.where(r_obs_spec<0,-np.inf,prob_1_spec)
            if trunc_zL:
                # prob_2 = dist.TruncatedNormal(zL, zL_sigma,low=0).log_prob(zL_obs)
                # prob_2 = jnp.where(zL_obs<0,-np.inf,prob_2)
                prob_2 = jax_truncnorm.logpdf(x=zL_obs,loc=zL, scale=zL_sigma,a=-zL/zL_sigma,b=np.inf)
            else:
                prob_2 = dist.Normal(zL, zL_sigma).log_prob(zL_obs)
            if trunc_zS:
                # prob_3 = dist.TruncatedNormal(zS, zS_sigma,low=0).log_prob(zS_obs)
                # prob_3 = jnp.where(zS_obs<0,-np.inf,prob_3)
                prob_3 = jax_truncnorm.logpdf(x=zS_obs,loc=zS, scale=zS_sigma,a=-zS/zS_sigma,b=np.inf)
            else:
                prob_3 = dist.Normal(zS, zS_sigma).log_prob(zS_obs)
            # Including the dependence of the lens redshift on the source redshifts (and that zL<zS in all cases):
            if lognorm_parent:
                s_z_min = 0.05
                s_z = s_c+s_m*zL # Assume a linear dependence of log-normal distribution on lens redshift
                sc_z = scale_c+scale_m*zL
                s_z = jnp.where(s_z<s_z_min,s_z_min,s_z)
                prob_4 = jax_lognormal(x=zS-zL,
                                        s=s_z,
                                        loc=0, #Defined to be 0 
                                        scale=sc_z).log_prob()
                # jax.debug.print('LIKELIHOOD Components: {a},{b},{c},{d}',
                #                 a=jnp.sum(prob_1),b=jnp.sum(prob_2),
                #                 c=jnp.sum(prob_3),d=jnp.sum(prob_4))
                if batch_bool: prob = prob_1+prob_2+prob_3+prob_4
                else: prob  = prob_1 + prob_2 + prob_3 + prob_4  #NOT Including spectra array, assumed empty.    
                # else: prob  = jnp.sum(prob_1)+jnp.sum(prob_1_spec)+jnp.sum(prob_2)+jnp.sum(prob_3)+jnp.sum(prob_4)             
            if not no_parent:
                if GMM_zL:
                    prob_4 = dist.Mixture(dist.Categorical(probs=jnp.array([w_zL,1-w_zL])),
                                [dist.TruncatedNormal(mu_zL_g_L_A, sigma_zL_g_L_A,low=0),
                                dist.TruncatedNormal(mu_zL_g_L_B, sigma_zL_g_L_B,low=0)]).log_prob(zL)+\
                            dist.Normal(zL, zL_sigma).log_prob(zL_obs)
                else:
                    prob_4 = dist.Normal(mu_zL_g_L, sigma_zL_g_L).log_prob(zL)
                    # prob_4 = jnp.where(zL<0,-np.inf,prob_4)s
                if GMM_zS:
                    # CHANGING THIS TO 0 - NO LONGER ASSERTING ZL<ZS HERE:
                    prob_5 = dist.Mixture(dist.Categorical(probs=jnp.array([w_zS,1-w_zS])),
                                [dist.TruncatedNormal(mu_zS_g_L_A, sigma_zS_g_L_A,low=0),
                                dist.TruncatedNormal(mu_zS_g_L_B, sigma_zS_g_L_B,low=0)]).log_prob(zS)+\
                            dist.Normal(zS, zS_sigma).log_prob(zS_obs)
                else:
                    prob_5 = dist.Normal(mu_zS_g_L, sigma_zS_g_L).log_prob(zS)
                    # prob_5 = jnp.where(zS<zL,-np.inf,prob_5)
                # jax.debug.print('NB: Including parent distribution in likelihood')
                if batch_bool: prob = prob_1+prob_2+prob_3+prob_4+prob_5 
                else: prob  = jnp.sum(prob_1)+jnp.sum(prob_1_spec)+jnp.sum(prob_2)+jnp.sum(prob_3)+jnp.sum(prob_4)+jnp.sum(prob_5)
            else:
                if batch_bool: prob = prob_1+prob_2+prob_3
                else: prob = jnp.sum(prob_1)+jnp.sum(prob_1_spec)+jnp.sum(prob_2)+jnp.sum(prob_3) 
            # jax.debug.print('Len 1 {a}',a=len(prob)) #Length = 1
            #NOTE: Do NOT comment out the next four lines unless absolutely sure Ode cannot be outside of (0,1):
            if batch_bool:
                prob = jnp.where(Ode*jnp.ones(len(prob))<0,-np.inf,prob)
                prob = jnp.where(Ode*jnp.ones(len(prob))>1,-np.inf,prob)
            else:                
                prob = jnp.where(Ode<0,-np.inf,prob)
                prob = jnp.where(Ode>1,-np.inf,prob)
            # jax.debug.print('Len 1.5 {a}',a=len(prob)) #length = 1
            # TEMPORARILY REMOVING THIS, AS ASSERTING ZS>ZL ANYWAY SO R_THEORY SHOULD BE >0 ALWAYS.
            # If I don't remove it, note that prob suddenly becomes of length N_sys, so becomes a different value (I.e. N_sys x larger)
            # prob = jnp.where(r_theory<0,-np.inf,prob) #Adding this in to see if fixes nan issue?
            # jax.debug.print('Len 2 {a}',a=len(prob)) #length = N(sys) if keep above line in. 
            if no_parent:
                pass
                # jax.debug.print('LIKELIHOOD Components: {a},{b},{c},{d},{e}',a=jnp.sum(prob_1),b=jnp.sum(prob_1_spec),c=jnp.sum(prob_2),d=jnp.sum(prob_3),e=jnp.sum(prob))
                # jax.debug.print('OM, Ode, Ok, w0, wa: {a},{b},{c},{d},{e}',a=OM,b=Ode,c=Ok,d=w,e=wa)
                # jax.debug.print('Min Redshifts and r: {a},{b} {c}',a=jnp.min(zL),b=jnp.min(zS),c=jnp.min(r_theory))
                # jax.debug.print('r_theory: {a}',a=jnp.sum(r_theory))
                # jax.debug.print('zL: {a}',a=jnp.sum(zL))
                # jax.debug.print('zS: {a}',a=jnp.sum(zS))
                # pass
            else:
                jax.debug.print('LIKELIHOOD Components: {a},{b},{c},{d},{e}',a=prob_1,b=prob_2,c=prob_3,d=prob_4,e=prob_5)
                jax.debug.print('OM, Ode, Ok, w0, wa: {a},{b},{c},{d},{e}',a=OM,b=Ode,c=Ok,d=w,e=wa)
            # jax.debug.print('Len 3 {a}',a=len(prob))
            if batch_bool: return prob
            # else: return jnp.sum(prob)
            else: return prob
        if batch_bool:
            assert len(spec_indx)==0 #Likelihood function doesn't include spectroscopic results when doing batching.
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
                prob = photometric_likelihood(batch_r_obs,batch_sigma_r_obs,batch_r_theory,
                                    jnp.array([]),jnp.array([]),jnp.array([]),
                                   batch_zL_obs,batch_zS_obs,
                                   batch_zL_sigma,batch_zS_sigma,
                                   batch_zL,batch_zS,
                                   mu_zL_g_L=mu_zL_g_L,mu_zS_g_L = mu_zS_g_L,
                                   w_zL=w_zL,w_zS=w_zS,
                                   mu_zL_g_L_A=mu_zL_g_L_A,mu_zL_g_L_B = mu_zL_g_L_B,
                                   sigma_zL_g_L_A=sigma_zL_g_L_A,sigma_zL_g_L_B=sigma_zL_g_L_B,
                                   GMM_zL=GMM_zL,GMM_zS=GMM_zS)
                if likelihood_check: return prob
                L = numpyro.factor("Likelihood",prob)
                # jax.debug.print('Checking if finite')
                # breakpoint_if_nonfinite(prob,zL,zS,r_theory,OM,Ode,Ok,w,wa)
        else:
            assert len(spec_indx)==0 #Otherwise will need to change likelihood to summing, so it includes spectra. But trying to not do this to reduce rounding errors.
            prob = photometric_likelihood(
                                   r_obs[phot_indx],sigma_r_obs[phot_indx],r_theory[phot_indx],
                                   r_obs[spec_indx],sigma_r_obs[spec_indx],r_theory[spec_indx],
                                   zL_obs[phot_indx],zS_obs[phot_indx],
                                   zL_sigma[phot_indx],zS_sigma[phot_indx],
                                   zL[phot_indx],zS[phot_indx],
                                   mu_zL_g_L=mu_zL_g_L,mu_zS_g_L = mu_zS_g_L,
                                   w_zL=w_zL,w_zS=w_zS,
                                   mu_zL_g_L_A=mu_zL_g_L_A,mu_zL_g_L_B = mu_zL_g_L_B,
                                   sigma_zL_g_L_A=sigma_zL_g_L_A,sigma_zL_g_L_B=sigma_zL_g_L_B,
                                   GMM_zL=GMM_zL,GMM_zS=GMM_zS)
            if likelihood_check: return prob
            L = numpyro.factor("Likelihood",prob)
    elif photometric and contaminated:
        print('Assuming contaminated, with photometric redshifts')
        # print('NOTE: Need to come up with a test function (e.g. known likelihood) to see what this is actually doing, rather than just\n'+\
            # ' assuming that because it gives the right answer it must be correct.')
        if not no_parent:
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
        # alpha_mu = jnp.squeeze(numpyro.sample("alpha_mu", dist.Uniform(0,2),sample_shape=(1,),rng_key=key))
        # alpha_scale = jnp.squeeze(numpyro.sample("alpha_scale", dist.Uniform(0.01,5),sample_shape=(1,),rng_key=key))
        # alpha_s = np.nan
        # alpha_mu_2 = jnp.squeeze(numpyro.sample("alpha_mu_2", dist.Uniform(0,2),sample_shape=(1,),rng_key=key))
        # alpha_scale_2 = jnp.squeeze(numpyro.sample("alpha_scale_2", dist.Uniform(0.01,5),sample_shape=(1,),rng_key=key))
        # alpha_w = jnp.squeeze(numpyro.sample("alpha_w", dist.Uniform(0,1),sample_shape=(1,),rng_key=key))
        def photometric_and_contaminated_likelihood(r_obs,sigma_r_obs,r_theory,
                                            # r_obs_spec,sigma_r_obs_spec,r_theory_spec, NOT Implemented yet
                                            zL_obs,zS_obs,
                                            zL_sigma,zS_sigma,
                                            zL,zS,
                                            mu_zL_g_L=None,mu_zS_g_L = None,
                                            w_zL=None,w_zS=None,
                                            mu_zL_g_L_A=None,mu_zL_g_L_B = None,
                                            sigma_zL_g_L_A=None,sigma_zL_g_L_B=None,
                                            GMM_zL=False,GMM_zS=False,
                                            P_tau=None,sigma_r_obs_2=None,
                                            mu_zL_g_NL=None,mu_zS_g_NL=None,
                                            sigma_zL_g_NL=None,sigma_zS_g_NL=None,
                                            sigma_01_g_L=None,sigma_10_g_L=None,
                                            sigma_01_g_NL=None,sigma_10_g_NL=None):
            #assert False #Where am I truncating zS in the likelihood - at 0 or at zL?
            prob_1a = jax_truncnorm.logpdf(x=r_obs,loc=r_theory, scale=sigma_r_obs,a=-r_theory/sigma_r_obs,b=np.inf)
            # prob_1a = dist.TruncatedNormal(r_theory, sigma_r_obs,low = 0).log_prob(r_obs)
            # prob_1a = jnp.where(r_obs<0,-np.inf,prob_1a)
            # jax.debug.print('Prob 1b: {r_obs},{r_theory},{sigma_r_obs_2}',r_obs=r_obs,r_theory=r_theory,sigma_r_obs_2=sigma_r_obs_2)
            prob_1b = dist.Mixture(dist.Categorical(jnp.array([simplex_sample.T[0],
                                                            simplex_sample.T[1],
                                                            simplex_sample.T[2]
                                                            ]).T),
                            [
                            dist.TruncatedNormal(alpha_mu_dict[0],alpha_scale_dict[0],low=0),
                            dist.TruncatedNormal(alpha_mu_dict[1],alpha_scale_dict[1],low=0),
                            dist.TruncatedNormal(alpha_mu_dict[2],alpha_scale_dict[2],low=0)
                             ]).log_prob(r_obs)
            # prob_1b = dist.Mixture(dist.Categorical(jnp.array([alpha_w,(1-alpha_w)]).T),
            #     [dist.TruncatedNormal(alpha_mu, alpha_scale,low=0),
            #      dist.TruncatedNormal(alpha_mu_2,alpha_scale_2,low=0)
            #         ]).log_prob(r_obs)
            # prob_1b = jax_truncnorm.logpdf(x=r_obs,loc=r_theory, scale=sigma_r_obs_2,
            #                                 a=-r_theory/sigma_r_obs_2,b=np.inf)
            # prob_1b = jax_truncnorm.logpdf(x = r_obs,loc = alpha_mu, scale = alpha_sigma,
            #                                a = -alpha_mu/alpha_sigma,b = np.inf)
            # prob_1b = dist.TruncatedNormal(r_theory, sigma_r_obs_2, low = 0).log_prob(r_obs)
            # prob_1b = jnp.where(r_obs<0,-np.inf,prob_1b)
            if trunc_zL:
                # prob_2 = dist.TruncatedNormal(zL, zL_sigma, low = 0).log_prob(zL_obs)
                # prob_2 = jnp.where(zL_obs<0,-np.inf,prob_2)
                prob_2 = jax_truncnorm.logpdf(x=zL_obs,loc=zL, scale=zL_sigma,a=-zL/zL_sigma,b=np.inf)
            else:
                prob_2 = dist.Normal(zL, zL_sigma).log_prob(zL_obs)
            if trunc_zS:
                # prob_3 = dist.TruncatedNormal(zS, zS_sigma, low = 0).log_prob(zS_obs)
                # prob_3 = jnp.where(zS_obs<0,-np.inf,prob_3)
                prob_3 = jax_truncnorm.logpdf(x=zS_obs,loc=zS, scale=zS_sigma,a=-zS/zS_sigma,b=np.inf)
            else:
                prob_3 = dist.Normal(zS, zS_sigma).log_prob(zS_obs)
            if not no_parent:
                assert False #Haven't checked that this implementation is up to date with the photometric likelihood function.
                if GMM_zL:
                    assert False #Not yet implemented
                else:
                    prob_4a = dist.TruncatedNormal(mu_zL_g_L,sigma_zL_g_L,low=0).log_prob(zL)
                    prob_4a = jnp.where(zL<0,-np.inf,prob_4a)
                    prob_4b = dist.TruncatedNormal(mu_zL_g_NL,sigma_zL_g_NL,low=0).log_prob(zL)
                    prob_4b = jnp.where(zL<0,-np.inf,prob_4b)
                if GMM_zS:
                    assert False #Not yet implemented
                else:
                    prob_5a = dist.TruncatedNormal(mu_zS_g_L,sigma_zS_g_L,low=0).log_prob(zS)
                    prob_5a = jnp.where(zS<0,-np.inf,prob_5a)
                    prob_5b = dist.TruncatedNormal(mu_zS_g_NL,sigma_zS_g_NL,low=zL).log_prob(zS)
                    prob_5b = jnp.where(zS<zL,-np.inf,prob_5b)
                # prob_4a = MVN_samp(mu_zL_g_L,mu_zS_g_L,sigma_zL_g_L,sigma_zS_g_L,zL,zS,sigma_01_g_L,sigma_10_g_L)
                # prob_4b = MVN_samp(mu_zL_g_NL,mu_zS_g_NL,sigma_zL_g_NL,sigma_zS_g_NL,zL,zS,sigma_01_g_NL,sigma_10_g_NL)
            '''
            Seems to be a problem with very small numbers - can cope if I increase the precision but still with only very small numbers of 
            systems => Problem fixed by having P_tau!=1.0 (even 0.9 fixed it).
            '''
            #'log_prob' finds the natural logarithm (not log10), hence these are natural-logged:
            if lognorm_parent:
                s_z_min = 0.05
                s_z = s_c+s_m*zL
                sc_z = scale_c+scale_m*zL
                s_z = jnp.where(s_z<s_z_min,s_z_min,s_z)
                prob_zL_zS = jax_lognormal(x=zS-zL,
                                        s=s_z,
                                        loc=0, #Defined to be 0 
                                        scale=sc_z).pdf()
            else: prob_zL_zS=1
            if no_parent: 
                # jax.debug.print('Likelihood Outputs: {a},{b},{c},{d} {e}',a=jnp.sum(prob_1a),b=jnp.sum(prob_1b),c=jnp.sum(prob_2),d=jnp.sum(prob_3),e=jnp.sum(prob_zL_zS))
                # jax.debug.print('P_tau,{P_tau}',P_tau=P_tau)
                prob =  likelihood_PC_no_parent(P_tau,prob_1a,prob_1b,prob_2,prob_3,prob_zL_zS) #prob_zL_zS is not logged
            else: 
                # jax.debug.print('Likelihood Outputs: {a},{b},{c},{d},{e},{f},{g},{h}',
                                # a=jnp.sum(prob_1a),b=jnp.sum(prob_1b),c=jnp.sum(prob_2),
                                # d=jnp.sum(prob_3),e=jnp.sum(prob_4a),f=jnp.sum(prob_4b),g=jnp.sum(prob_5a),h=jnp.sum(prob_5b))
                prob =  likelihood_PC(P_tau,prob_1a,prob_1b,prob_2,prob_3,prob_4a,prob_4b,prob_5a,prob_5b)
            # jax.debug.print('Likelihood: {p}',p=jnp.sum(prob))
            prob = jnp.where(Ode*jnp.ones(len(prob))<0,-np.inf,prob)           
            # jax.debug.print('Likelihood {p}',p=jnp.sum(prob))
            prob = jnp.where(Ode*jnp.ones(len(prob))>1,-np.inf,prob)
            # jax.debug.print('Likelihood {p}',p=jnp.sum(prob))
            # jax.debug.print('Likelihood {p}',p=prob)
            # jax.debug.print('OM: {a}',a=OM)
            return prob                                             
        if batch_bool:
            with numpyro.plate("N", zL_obs.shape[0], subsample_size = subsample_size):
                batch_r_obs = numpyro.subsample(r_obs,event_dim=0)
                batch_sigma_r_obs = numpyro.subsample(sigma_r_obs,event_dim=0)
                batch_sigma_r_obs_2 = numpyro.subsample(sigma_r_obs_2,event_dim=0)
                batch_r_theory = numpyro.subsample(r_theory,event_dim=0)
                batch_zL_obs = numpyro.subsample(zL_obs,event_dim=0)
                batch_zS_obs = numpyro.subsample(zS_obs,event_dim=0)
                batch_zL_sigma = numpyro.subsample(zL_sigma,event_dim=0)
                batch_zS_sigma = numpyro.subsample(zS_sigma,event_dim=0)
                batch_zL = numpyro.subsample(zL,event_dim=0)
                batch_zS = numpyro.subsample(zS,event_dim=0) 
                batch_P_tau = numpyro.subsample(P_tau,event_dim=0)
                prob = photometric_and_contaminated_likelihood(batch_r_obs,batch_sigma_r_obs,batch_r_theory,
                                # jnp.array([]),jnp.array([]),jnp.array([]), NOT Implemented yet
                                batch_zL_obs,batch_zS_obs,
                                batch_zL_sigma,batch_zS_sigma,
                                batch_zL,batch_zS,
                                mu_zL_g_L=mu_zL_g_L,mu_zS_g_L = mu_zS_g_L,
                                w_zL=w_zL,w_zS=w_zS,
                                mu_zL_g_L_A=mu_zL_g_L_A,mu_zL_g_L_B = mu_zL_g_L_B,
                                sigma_zL_g_L_A=sigma_zL_g_L_A,sigma_zL_g_L_B=sigma_zL_g_L_B,
                                GMM_zL=GMM_zL,GMM_zS=GMM_zS,
                                P_tau=batch_P_tau,sigma_r_obs_2=batch_sigma_r_obs_2,
                                mu_zL_g_NL=mu_zL_g_NL,mu_zS_g_NL=mu_zS_g_NL,
                                sigma_zL_g_NL=sigma_zL_g_NL,sigma_zS_g_NL=sigma_zS_g_NL,
                                sigma_01_g_L=sigma_01_g_L,sigma_10_g_L=sigma_10_g_L,
                                sigma_01_g_NL=sigma_01_g_NL,sigma_10_g_NL=sigma_10_g_NL)
        else:
            prob = photometric_and_contaminated_likelihood(r_obs[phot_indx],sigma_r_obs[phot_indx],r_theory[phot_indx],
                                            # r_obs[spec_indx],sigma_r_obs[spec_indx],r_theory[spec_indx], NOT Implemented yet
                                            zL_obs[phot_indx],zS_obs[phot_indx],
                                            zL_sigma[phot_indx],zS_sigma[phot_indx],
                                            zL[phot_indx],zS[phot_indx],
                                            mu_zL_g_L=mu_zL_g_L,mu_zS_g_L = mu_zS_g_L,
                                            w_zL=w_zL,w_zS=w_zS,
                                            mu_zL_g_L_A=mu_zL_g_L_A,mu_zL_g_L_B = mu_zL_g_L_B,
                                            sigma_zL_g_L_A=sigma_zL_g_L_A,sigma_zL_g_L_B=sigma_zL_g_L_B,
                                            GMM_zL=GMM_zL,GMM_zS=GMM_zS,
                                            P_tau=P_tau,sigma_r_obs_2=sigma_r_obs_2,
                                            mu_zL_g_NL=mu_zL_g_NL,mu_zS_g_NL=mu_zS_g_NL,
                                            sigma_zL_g_NL=sigma_zL_g_NL,sigma_zS_g_NL=sigma_zS_g_NL,
                                            sigma_01_g_L=sigma_01_g_L,sigma_10_g_L=sigma_10_g_L,
                                            sigma_01_g_NL=sigma_01_g_NL,sigma_10_g_NL=sigma_10_g_NL)
        if likelihood_check: return prob
        return prob
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
                prob = jnp.where(Ode*jnp.ones(len(prob))<0,-np.inf,prob)
                prob = jnp.where(Ode*jnp.ones(len(prob))>1,-np.inf,prob) 
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
                prob = jnp.where(Ode<0,-np.inf,prob)
                prob = jnp.where(Ode>1,-np.inf,prob) 
                numpyro.factor('Likelihood',prob)    
        # numpyro.sample("r", dist.Normal(r_theory, sigma_r_obs), obs=r_obs)

def j_likelihood_SL(zL_obs,zS_obs,sigma_zL_obs,sigma_zS_obs,r_obs,sigma_r_obs,sigma_r_obs_2=[np.nan],P_tau_0 = [],cosmo_type='',
                    photometric=False,contaminated=False,H0=np.nan,key=None,
                    likelihood_check=False,likelihood_dict = {},cov_redshift=False,early_return=False,
                    batch_bool=True, wa_const = False, w0_const = False,GMM_zL = False,GMM_zS = False,
                    fixed_GMM = False, GMM_zL_dict = {}, GMM_zS_dict = {},spec_indx = [],no_parent=False,
                    trunc_zL=False,trunc_zS=False,P_tau_dist = False,sigma_P_tau = [],lognorm_parent = False,
                    unimodal_beta=True,bimodal_beta=False,true_zL_zS_dep = False,batch_indx_array=[],N_dim=1
                    ):
    
    N_batch = len(batch_indx_array)
    prior_dict = {'OM':None,'Ode':None,'Ok':None,'w':None,'wa':None,'s_m':None,'s_c':None,'scale_m':None,'scale_c':None,
                  'alpha_mu_dict':None,'alpha_scale_dict':None,'simplex_sample':None}
    # Prior on Omega_M:
    OM = jnp.squeeze(numpyro.sample("OM", dist.Uniform(0,1),sample_shape=(1,),rng_key=key))
    if cosmo_type in ['FlatLambdaCDM','FlatwCDM']:
        #Don't care about Ode, as it isn't an argument for the cosmology (OM and Ok are instead)
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
        w = jnp.squeeze(numpyro.sample("w", dist.Uniform(-6,4),sample_shape=(1,),rng_key=key)) #Physicality constraints, (-6,4)
        wa = numpyro.deterministic('wa',0.0)
    elif wa_const == False and w0_const == True:
        print('Assuming an evolving dark energy equation of state, but with w0 fixed at -1.')
        w = numpyro.deterministic('w',-1.0)
        wa = jnp.squeeze(numpyro.sample("wa", dist.Uniform(-3,1),sample_shape=(1,),rng_key=key)) #Matching Tian's constraints for now
    else:
        print('Assuming non-trivial dark energy equation of state')
        w = jnp.squeeze(numpyro.sample("w", dist.Uniform(-3,1),sample_shape=(1,),rng_key=key)) # Adding tighter constraints for now. Previously Physicality constraints, (-6,4)
        wa = jnp.squeeze(numpyro.sample("wa", dist.Uniform(-3,1),sample_shape=(1,),rng_key=key)) #Matching Tian's constraints for now
    if lognorm_parent:
        if true_zL_zS_dep:
            print('Using true P(zL|zS) dependence in likelihood')
            s_m = -0.22745602;s_c =  0.61472073
            scale_m = 1.15897213;scale_c = 0.89219335
        else:
            #These MUST be uniform distributions if using kde to combine batches of posteriors:
            s_m = jnp.squeeze(numpyro.sample("s_m", dist.Uniform(-1,0),sample_shape=(1,),rng_key=key)) #-0.2
            #These MUST be uniform distributions if using kde to combine batches of posteriors:
            s_c = jnp.squeeze(numpyro.sample("s_c", dist.Uniform(0.01,2),sample_shape=(1,),rng_key=key)) #0.6
            #These MUST be uniform distributions if using kde to combine batches of posteriors:
            scale_m = jnp.squeeze(numpyro.sample("scale_m", dist.Uniform(0,6),sample_shape=(1,),rng_key=key)) #1.0
            #These MUST be uniform distributions if using kde to combine batches of posteriors:
            scale_c =  jnp.squeeze(numpyro.sample("scale_c", dist.Uniform(0.1,5),sample_shape=(1,),rng_key=key)) #1.0
    N_comp = 3
    alpha_mu_dict = {elem:numpyro.sample(f'alpha_mu_{elem}',dist.Uniform(0,2),sample_shape=(1,),rng_key=key) for elem in range(N_comp)}
    alpha_scale_dict = {elem:numpyro.sample(f'alpha_scale_{elem}',dist.LogUniform(0.01,5),sample_shape=(1,),rng_key=key) for elem in range(N_comp)}
    simplex_sample = numpyro.sample('alpha_weights',dist.Dirichlet(concentration=jnp.array([1.0]*N_comp)),rng_key=key)
    #
    prior_dict = {'OM':OM,'Ode':Ode,'Ok':Ok,'w':w,'wa':wa,'s_m':s_m,'s_c':s_c,'scale_m':scale_m,'scale_c':scale_c,
                  'alpha_mu_dict':alpha_mu_dict,'alpha_scale_dict':alpha_scale_dict,'simplex_sample':simplex_sample}
    total_prob = 0
    for batch_number in range(N_batch):
        batch_indx_i = batch_indx_array[batch_number]
        batch_prob_i = j_likelihood_SL_batch(
                    zL_obs[batch_indx_i],zS_obs[batch_indx_i],
                    sigma_zL_obs[batch_indx_i],sigma_zS_obs[batch_indx_i],
                    r_obs[batch_indx_i],sigma_r_obs[batch_indx_i],
                    sigma_r_obs_2=[sigma_r_obs_2],
                    P_tau_0 = P_tau_0[batch_indx_i],cosmo_type=cosmo_type,
                    photometric=photometric,contaminated=contaminated,H0=H0,key=key,
                    likelihood_check=likelihood_check,likelihood_dict = likelihood_dict,cov_redshift=cov_redshift,early_return=early_return,
                    batch_bool=batch_bool, wa_const = wa_const, w0_const = w0_const,GMM_zL = GMM_zL,GMM_zS = GMM_zS,
                    fixed_GMM = fixed_GMM, GMM_zL_dict = GMM_zL_dict, GMM_zS_dict = GMM_zS_dict,spec_indx = spec_indx,no_parent=no_parent,
                    trunc_zL=trunc_zL,trunc_zS=trunc_zS,P_tau_dist = P_tau_dist,
                    sigma_P_tau = sigma_P_tau[batch_indx_i],
                    lognorm_parent = lognorm_parent,
                    unimodal_beta=unimodal_beta,bimodal_beta=bimodal_beta,true_zL_zS_dep = true_zL_zS_dep,
                    prior_dict = prior_dict,batch_number=batch_number)
        total_prob += jnp.sum(batch_prob_i)
        del batch_prob_i
        gc.collect()
    total_prob-=jnp.sqrt(N_dim) #Rescaling the likelihood to help convergence.
    L = numpyro.factor("Likelihood",total_prob)

# Code to run the cosmology posterior sampling:
def run_MCMC(photometric,contaminated,cosmo_type,
            zL_obs,zS_obs,sigma_zL_obs,sigma_zS_obs,
            r_obs,sigma_r_obs,sigma_r_obs_2,P_tau_0,
            num_warmup = 200,num_samples=1000,num_chains=2,
            H0=np.nan,target_accept_prob=0.8,cov_redshift=False,warmup_file=np.nan,
            batch_bool=True,wa_const=False,w0_const=False,GMM_zL = False,GMM_zS = False,key_int = 0,
            fixed_GMM=False,GMM_zL_dict={},GMM_zS_dict={},nested_sampling=False,zL_true=None,zS_true=None,
            no_parent=False,initialise_to_truth=False,trunc_zL=False,trunc_zS=False,
            P_tau_dist=False,sigma_P_tau = None,lognorm_parent=False,
            r_true = None,unimodal_beta=True,bimodal_beta=False,
            true_zL_zS_dep=False,N_batch=1):
    print('Random key:',key_int)
    # assert False #alpha_scale prior is LogUniform not Uniform, so batching is not ok (if I multiply by the prior multiple times)
    # jax.profiler.start_trace("./memory_profiling")
    batch_indx_array = jnp.array_split(jnp.arange(len(zL_obs)),N_batch)
    if unimodal_beta:
        print('USING MAXIMUM (AND POSSIBLY VARYING) SIGMA_P_TAU POSSIBLE')
        sigma_P_tau = beta_class().max_sigma_for_unimodal_beta(P_tau_0)
    elif bimodal_beta:
        print('USING SIGMA_P_TAU CORRESPONDING TO BIMODAL BETA')
        sigma_P_tau = beta_class().min_sigma_for_bimodal_beta(P_tau_0)
    model_args = {'zL_obs':zL_obs,'zS_obs':zS_obs,
                'sigma_zL_obs':sigma_zL_obs,'sigma_zS_obs':sigma_zS_obs,
                'r_obs':r_obs,'sigma_r_obs':sigma_r_obs,'sigma_r_obs_2':sigma_r_obs_2,
                'P_tau_0':P_tau_0,'cosmo_type':cosmo_type,
                'photometric':photometric,'contaminated':contaminated,
                'H0':H0,'cov_redshift':cov_redshift,'batch_bool':batch_bool,
                'wa_const':wa_const,'w0_const':w0_const,'GMM_zL':GMM_zL,'GMM_zS':GMM_zS,
                'GMM_zL_dict':GMM_zL_dict,'GMM_zS_dict':GMM_zS_dict,'fixed_GMM':fixed_GMM,
                'no_parent':no_parent,'trunc_zL':trunc_zL,'trunc_zS':trunc_zS,
                'P_tau_dist':P_tau_dist,'sigma_P_tau':sigma_P_tau,'lognorm_parent':lognorm_parent,
                'unimodal_beta':unimodal_beta,'bimodal_beta':bimodal_beta,'true_zL_zS_dep':true_zL_zS_dep,
                'batch_indx_array':batch_indx_array,'N_dim':len(zL_obs)*3}
    print(f'Model args: {model_args}')
    key = jax.random.PRNGKey(key_int)
    print(f'Target Accept Prob: {target_accept_prob}')
    print(f'Batch bool: {batch_bool}')
    st = time.time()
    j_likelihood_SL(**model_args,key=key,early_return=True)
    # return
    mt=time.time()
    j_likelihood_SL(**model_args,key=key,early_return=True)
    et=time.time()
    print('Uncompiled time',mt-st)
    print('Compiled time',et-mt)
    #USEFUL LINK REGARDING SPEEDING UP NUTS AND HMCECS:
    #https://forum.pyro.ai/t/scalability-of-hmcecs/5349/12
    if initialise_to_truth:
        print('INITIALISING TO TRUE VALUE')
        assert not nested_sampling # Not set up
        init_strategy = init_to_value(values={'Ok':0.0,'OM':0.3,'w':-1.0,'wa':0.0,
                                                                'zL':zL_true,'zS':zS_true,'Ode':0.7,
                                        'alpha_s': 0.28042537528031597, 'alpha_mu': -0.31602834154215675, 'alpha_scale': 0.8026664235668167,
                                        'r_theory_2':jnp.array([r_true])
                                        })
    else:
        init_strategy = init_to_uniform
    if nested_sampling:
        print('USING NESTED SAMPLING')
        NS = NestedSampler(model=j_likelihood_SL,constructor_kwargs={'num_live_points':20})#,num_live_points=20,max_samples=num_samples)
        print('Running')
        NS.run(random.PRNGKey(2), **model_args)
        print('Getting samples')
        samples = NS.get_samples(random.PRNGKey(3), num_samples=1000)
    elif batch_bool:
        inner_kernel = NUTS(model = j_likelihood_SL,target_accept_prob = target_accept_prob,init_strategy=init_strategy)
        outer_kernel = HMCECS(inner_kernel, num_blocks=100)
    else:
        # print('USING DENSE MASS')
        outer_kernel =  NUTS(model = j_likelihood_SL,target_accept_prob = target_accept_prob,init_strategy=init_strategy)#,dense_mass=True)
    sampler_0 = MCMC(outer_kernel,
                    num_warmup=num_warmup,
                    num_samples=num_samples,
                    num_chains=num_chains,
                    progress_bar=True)
    print('Starting Warmup:')
    sampler_0.warmup(key,**model_args,collect_warmup=True)
    ##
    warmup_dict = JAX_samples_to_dict(sampler_0,separate_keys=True,cosmo_type=cosmo_type,wa_const=wa_const,w0_const=w0_const,
                                      fixed_GMM=fixed_GMM,N_sys_per_batch=[len(batch_indx_array[elem]) for elem in range(N_batch)])
    db_JAX_warmup = pd.DataFrame(warmup_dict)
    db_JAX_warmup.to_csv(warmup_file,index=False)
    print(f'Saved warmup to {warmup_file}')
    ##
    print("Starting main run:")
    N_split = 10
    for i in range(N_split):
        sampler_0.post_warmup_state = sampler_0.last_state
        sampler_0.run(sampler_0.post_warmup_state.rng_key,**model_args,key=None)
        raw_mcmc_samples = sampler_0.get_samples(group_by_chain=True)
        print(raw_mcmc_samples.keys())
        if i!=(N_split-1): #Temporarily deleting these to check it runs ok, though I should definitely save + include them in my chains.
            del raw_mcmc_samples
            gc.collect()
    # sampler_0.run(key,**model_args,key=None)
    print('Finished main run')
    # jax.profiler.stop_trace()
    return sampler_0,[len(batch_indx_array[elem]) for elem in range(N_batch)]
    # Without HMCECS:
    # sampler_0 = infer.MCMC(
    #     infer.NUTS(model = j_likelihood_SL,
    #                target_accept_prob = target_accept_prob),
    #     num_warmup=num_warmup,
    #     num_samples=num_samples,
    #     num_chains=num_chains,
    #     progress_bar=True)
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


'''
addqueue -q cmbgpu --gpus 4 --gputype rtx3090with24gb -m 59 /mnt/users/hollowayp/python114_archive/bin/python3.11 ./run_zBEAMS_JAX.py --filein ./databases/real_paltas_population_TP_100000_FP_100000_Spec_10000_P_0.5_extrem.csv  --c True --p True --cosmo wCDM --num_chains 1 --target 0.99 --num_samples 10 --num_warmup 10 --no_parent --batch False  --P_tau_dist --sigma_P_tau 0.2 --no_parent --trunc_zS --trunc_zL --lognorm_parent --bimodal_beta False --unimodal_beta True --key 0 --batch_version --N_batch 10

Works for 100k (not OOM at least): - DO NOT EDIT.
addqueue -q cmbgpu --gpus 4 --gputype rtx3090with24gb -m 59 /mnt/users/hollowayp/python114_archive/bin/python3.11 ./run_zBEAMS_JAX.py --filein ./databases/real_paltas_population_TP_100000_FP_100000_Spec_10000_P_0.5_extrem.csv  --c True --p True --cosmo wCDM --num_chains 1 --target 0.99 --num_samples 10 --num_warmup 10 --no_parent --batch False  --P_tau_dist --sigma_P_tau 0.2 --no_parent --trunc_zS --trunc_zL --lognorm_parent --bimodal_beta False --unimodal_beta True --key 0 --batch_version --N_batch 10
'''

