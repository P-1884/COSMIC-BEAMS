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
def likelihood_PC_no_parent(P_tau,prob_1a,prob_1b,prob_2,prob_3,prob_zL_zS,prob_FP_zL,prob_FP_zS): #prob_zL_zS is not logged
        # jax.debug.print('Likelihood_pc Output: {a},{b},{c},{d},{e},{f},{g}',a=P_tau,b=jnp.exp(prob_1a),c=(1-P_tau),d=jnp.exp(prob_1b),e=prob_2,f=prob_3,g=prob_zL_zS)
        # jax.debug.print('To be logged {a}',a=jnp.min(P_tau*jnp.exp(prob_1a)+(1-P_tau)*jnp.exp(prob_1b)))
        likelihood_p1 = P_tau*jnp.exp(prob_1a+prob_2+prob_3)*prob_zL_zS
        likelihood_p2 = (1-P_tau)*jnp.exp(prob_1b+prob_FP_zL+prob_FP_zS)
        likelihood_prelim = jnp.log(likelihood_p1+likelihood_p2)
        #Accounting for rounding errors - when p1 is zero, can just use the logged version of p2 alone, and vice versa.
        likelihood_prelim = jnp.where(likelihood_p1==0,jnp.log(1-P_tau)+prob_1b+prob_FP_zL+prob_FP_zS,likelihood_prelim)
        likelihood_prelim = jnp.where(likelihood_p2==0,jnp.log(P_tau)+prob_1a+prob_2+prob_3+jnp.log(prob_zL_zS),likelihood_prelim)
        ### PROB_2 AND PROB_3 SHOULD MAYBE NOT BE HERE???
        return likelihood_prelim

def j_likelihood_SL(zL_obs,zS_obs,sigma_zL_obs,sigma_zS_obs,r_obs,sigma_r_obs,sigma_r_obs_2=[np.nan],P_tau_0 = [],cosmo_type='',
                    photometric=False,contaminated=False,H0=np.nan,key=None,
                    likelihood_check=False,likelihood_dict = {},cov_redshift=False,early_return=False,
                    batch_bool=True, wa_const = False, w0_const = False,GMM_zL = False,GMM_zS = False,
                    fixed_GMM = False, GMM_zL_dict = {}, GMM_zS_dict = {},spec_indx = [],no_parent=False,
                    trunc_zL=False,trunc_zS=False,P_tau_dist = False,sigma_P_tau = [],lognorm_parent = False,
                    unimodal_beta=True,bimodal_beta=False,true_zL_zS_dep = False,fixed_alpha=False,alpha_dict = {},
                    P_tau_regularisation=False,P_tau_regularisation_factor=jnp.nan,
                    fixed_beta_gamma=False,beta_dict = {},gamma_dict = {}):
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
    if fixed_alpha: assert len(alpha_dict)>0;assert photometric and contaminated;print('Using fixed FP parent distribution') #Other cases not yet implemented
    if lognorm_parent: assert not GMM_zL and not GMM_zS; assert no_parent
    if P_tau_dist: assert isinstance(sigma_P_tau,float) or len(sigma_P_tau)==len(P_tau_0)
    N_lens_expect = jnp.sum(P_tau_0)
    print(f'Number of expected lenses: {N_lens_expect}')
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
            zL = numpyro.sample('zL',dist.Uniform(low = zL_obs_low_lim,high = zL_obs_up_lim),sample_shape=(1,),rng_key=key).flatten()
            ##Minimum value is zL. First term: zS if zS>0, else 0. Second term, zL if zS<zL, else 0
            zS_obs_low_lim = jnp.array(zS_obs-5*zS_sigma)
            zS_obs_low_lim = zS_obs_low_lim*(zS_obs_low_lim>zL) + zL*(zS_obs_low_lim<zL)
            zS_obs_low_lim = zS_obs_low_lim*(zS_obs_low_lim>0)
            zS_obs_up_lim = jnp.array(zS_obs+5*zS_sigma)
            zS = numpyro.sample('zS',dist.Uniform(low = zS_obs_low_lim, high = zS_obs_up_lim),sample_shape=(1,),rng_key=key).flatten()
            #
            # Assumes a lognormal relation for P(zS|zL), with a linear dependence of the lognormal hyperparameters on redshift:
            if lognorm_parent:
                if true_zL_zS_dep:
                    print('Using true P(zL|zS) dependence in likelihood')
                    s_m = -0.22745602 
                    s_c =  0.61472073
                    scale_m = 1.15897213 
                    scale_c = 0.89219335
                else:
                    #These MUST be uniform distributions if using kde to combine batches of posteriors:
                    s_m = jnp.squeeze(numpyro.sample("s_m", dist.Uniform(-1,0),sample_shape=(1,),rng_key=key)) #-0.2
                    #These MUST be uniform distributions if using kde to combine batches of posteriors:
                    s_c = jnp.squeeze(numpyro.sample("s_c", dist.Uniform(0.01,2),sample_shape=(1,),rng_key=key)) #0.6
                    #These MUST be uniform distributions if using kde to combine batches of posteriors:
                    scale_m = jnp.squeeze(numpyro.sample("scale_m", dist.Uniform(0,6),sample_shape=(1,),rng_key=key)) #1.0
                    #These MUST be uniform distributions if using kde to combine batches of posteriors:
                    scale_c =  jnp.squeeze(numpyro.sample("scale_c", dist.Uniform(0.1,5),sample_shape=(1,),rng_key=key)) #1.0
            if not no_parent:
                if GMM_zL:
                    assert False
                else:
                    mu_zL_g_L = jnp.squeeze(numpyro.sample("mu_zL_g_L", dist.Uniform(0.1,1.5),sample_shape=(1,),rng_key=key))
                    sigma_zL_g_L = jnp.squeeze(numpyro.sample("sigma_zL_g_L", dist.Uniform(0.01,1),sample_shape=(1,),rng_key=key))
                if GMM_zS:
                    assert False
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
        # Prior on Omega_M:
        OM = jnp.squeeze(numpyro.sample("OM", dist.Uniform(0,1),sample_shape=(1,),rng_key=key))
        if cosmo_type in ['FlatLambdaCDM','FlatwCDM']:
            #Don't care about Ode, as it isn't an argument for the cosmology (OM and Ok are instead)
            print('Assuming a flat universe')
            Ok = numpyro.deterministic('Ok',0.0)
            Ode = numpyro.deterministic('Ode',1-(OM+Ok))
        else:
            print('Assuming the universe may have curvature')
            # OLD: Sampling OM and Ok uniformly. Ode is therefore sampled from a weird shaped distribution in the range (-1,2), but
            # # which is uniform in the range (0,1)
            # # This can therefore be cropped to a uniform distribution by setting the likelihood to -np.inf if Ode is outside of (0,1).
            # Ok = jnp.squeeze(numpyro.sample("Ok", dist.Uniform(-1,1),sample_shape=(1,),rng_key=key))
            # Ode = numpyro.deterministic('Ode',1-(OM+Ok))
            #Now just sampling Ode uniformly, and then setting Ok to 1-(OM+Ode):
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
    cosmo = jc.Cosmology(Omega_c=OM, h=H0/100, Omega_k=Ok, w0=w,
                         Omega_b=Ob, wa=wa, sigma8=s8, n_s=n_s)
    # Calculating the theoretical 'r' ratio for given lens/source redshifts and cosmology:
    if cosmo_type in ['FlatLambdaCDM','FlatwCDM']: r_theory = j_r_SL_flat(zL,zS,cosmo)
    else: r_theory = j_r_SL(zL,zS,cosmo)
    if early_return: return
    # If contaminated, need to know the prior probability each system is a lens (P_tau_0):
    if contaminated:
        if P_tau_dist: # Uses a distribution for P_tau, with mean P_tau_0 and width sigma_P_tau, rather than a single value:
            Beta_class_instance = beta_class(mean=P_tau_0,sigma=sigma_P_tau)
            beta_A = Beta_class_instance.A
            beta_B = Beta_class_instance.B
            P_tau = numpyro.sample('P_tau',dist.Beta(beta_A,beta_B),sample_shape = (1,),rng_key=key).flatten()
        else:
            P_tau = P_tau_0
        P_tau = P_tau.astype('float') #Needs to be a float for dist.Categorical to work
        # r_theory_2 = jnp.squeeze(numpyro.sample("r_theory_2", dist.TruncatedNormal(loc=alpha_mu,scale=alpha_sigma,
        #                                                                             low=0),sample_shape=(1,),rng_key=key))
    if photometric and contaminated:
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
        if fixed_alpha:
            N_comp = len(alpha_dict['mu'])
            alpha_mu_dict = {elem:numpyro.deterministic(f'alpha_mu_{elem}',alpha_dict['mu'][elem]) for elem in range(N_comp)}
            alpha_scale_dict = {elem:numpyro.deterministic(f'alpha_scale_{elem}',alpha_dict['scale'][elem]) for elem in range(N_comp)}
            simplex_sample = numpyro.deterministic('alpha_weights',alpha_dict['weights'])
        else:
            N_comp = 3
            alpha_mu_dict = {elem:numpyro.sample(f'alpha_mu_{elem}',dist.Uniform(0,2),sample_shape=(1,)) for elem in range(N_comp)}
            alpha_scale_dict = {elem:numpyro.sample(f'alpha_scale_{elem}',dist.Uniform(0.01,5),sample_shape=(1,)) for elem in range(N_comp)}
            simplex_sample = numpyro.sample('alpha_weights',dist.Dirichlet(concentration=jnp.array([1.0]*N_comp)))
        if True:
            if fixed_beta_gamma:
                beta_mu = numpyro.deterministic('beta_mu',beta_dict['mu'])
                beta_scale = numpyro.deterministic('beta_scale',beta_dict['scale'])
                gamma_mu = numpyro.deterministic('gamma_mu',gamma_dict['mu'])
                gamma_scale = numpyro.deterministic('gamma_scale',gamma_dict['scale'])
            else:
                #zL distribution for FP's:
                beta_mu = numpyro.sample('beta_mu',dist.Uniform(0,2),sample_shape=(1,))
                beta_scale = numpyro.sample('beta_scale',dist.Uniform(0.01,5),sample_shape=(1,))
                #zS distribution for FP's:
                gamma_mu = numpyro.sample('gamma_mu',dist.Uniform(0,5),sample_shape=(1,))
                gamma_scale = numpyro.sample('gamma_scale',dist.Uniform(0.01,5),sample_shape=(1,))
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
                # prob_4a = MVN_samp(mu_zL_g_L,mu_zS_g_L,sigma_zL_g_L,sigma_zS_g_L,zL,zS,sigma_01_g_L,sigma_10_g_L)
                # prob_4b = MVN_samp(mu_zL_g_NL,mu_zS_g_NL,sigma_zL_g_NL,sigma_zS_g_NL,zL,zS,sigma_01_g_NL,sigma_10_g_NL)
            if True:
                prob_FP_zL = jax_truncnorm.logpdf(x=zL,loc=beta_mu, scale=beta_scale,a=-beta_mu/beta_scale,b=np.inf)
                prob_FP_zS = jax_truncnorm.logpdf(x=zS,loc=gamma_mu, scale=gamma_scale,a=-gamma_mu/gamma_scale,b=np.inf)
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
                prob =  likelihood_PC_no_parent(P_tau,prob_1a,prob_1b,prob_2,prob_3,prob_zL_zS,prob_FP_zL,prob_FP_zS) #prob_zL_zS is not logged
            else: 
                prob =  likelihood_PC(P_tau,prob_1a,prob_1b,prob_2,prob_3,prob_4a,prob_4b,prob_5a,prob_5b)
            prob = jnp.where(Ode*jnp.ones(len(prob))<0,-np.inf,prob)           
            prob = jnp.where(Ode*jnp.ones(len(prob))>1,-np.inf,prob)
            if P_tau_regularisation:
                prob = prob + dist.Normal(loc=N_lens_expect,scale=P_tau_regularisation_factor*N_lens_expect).log_prob(jnp.sum(P_tau))
            return prob                                             
        if batch_bool:
            assert False
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
        L = numpyro.factor("Likelihood",prob)
 
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
            true_zL_zS_dep=False,alpha_dict={},fixed_alpha=False,
            fixed_beta_gamma=False,beta_dict={},gamma_dict={},
            P_tau_regular=False,P_tau_regular_factor=np.nan):
    print('Random key:',key_int)
    # assert False #Just a reminder - alpha_scale prior is LogUniform not Uniform, so batching is not ok.
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
                'alpha_dict':alpha_dict,'fixed_alpha':fixed_alpha,
                'fixed_beta_gamma':fixed_beta_gamma,'beta_dict':beta_dict,'gamma_dict':gamma_dict,
                'P_tau_regularisation':P_tau_regular,'P_tau_regularisation_factor':P_tau_regular_factor}
    print(f'Model args: {model_args}')
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
    if initialise_to_truth:
        assert False
    else:
        init_strategy = init_to_uniform
    if nested_sampling:
        assert False
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
    warmup_dict = JAX_samples_to_dict(sampler_0,separate_keys=True,cosmo_type=cosmo_type,wa_const=wa_const,w0_const=w0_const,fixed_GMM=fixed_GMM)
    db_JAX_warmup = pd.DataFrame(warmup_dict)
    db_JAX_warmup.to_csv(warmup_file,index=False)
    print(f'Saved warmup to {warmup_file}')
    print("Starting main run:")
    sampler_0.run(key,**model_args,key=None)
    return sampler_0
