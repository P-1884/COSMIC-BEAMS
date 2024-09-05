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
from jax.random import PRNGKey
import glob
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
                    photometric=False,contaminated=False,H0=np.nan,key=None,no_parent=False,
                    trunc_zL=False,trunc_zS=False,P_tau_dist = False,sigma_P_tau = [],lognorm_parent = False,
                    unimodal_beta=True,bimodal_beta=False,true_zL_zS_dep = False,
                    sampling_dict = {},
                    ):
    #Other permutations haven't yet been tested:
    assert photometric and contaminated and lognorm_parent
    OM = sampling_dict['OM'];Ok = sampling_dict['Ok'];Ode = sampling_dict['Ode'];w = sampling_dict['w'];wa = sampling_dict['wa']
    s_m = sampling_dict['s_m'];s_c = sampling_dict['s_c'];scale_m = sampling_dict['scale_m'];scale_c = sampling_dict['scale_c']
    alpha_mu_dict = sampling_dict['alpha_mu_dict'];alpha_scale_dict = sampling_dict['alpha_scale_dict']
    simplex_sample = sampling_dict['simplex_sample']
    zL = sampling_dict['zL']
    zS = sampling_dict['zS']
    P_tau = sampling_dict['P_tau']
    zL_sigma = sigma_zL_obs;zS_sigma = sigma_zS_obs
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
    if P_tau_dist: assert isinstance(sigma_P_tau,float) or len(sigma_P_tau)==len(P_tau_0)
    # Lens and source redshift parent hyperparameters (assuming gaussian mixture model):
    #assert False #Also the MCMC redshift-only inference should definitely match the best fit value right??
    s8 = 0.8;n_s = 0.96;Ob=0; #Putting all the matter in dark-matter (doesn't make a difference)
    # The priors on lens and source redshifts, as well as any population redshift hyperparameters for photometric systems are added below:
    if photometric:
        print('Assuming photometric redshifts')
        # Assumes a lognormal relation for P(zS|zL), with a linear dependence of the lognormal hyperparameters on redshift:
        # Adding a covariance term between zL and zS (this is no longer in use):
    else:
        # If photometric=False, the redshift measurements are assumed to be perfect, with zero uncertainty.
        print('Assuming spectroscopic redshifts')
        zL = zL_obs #Think still need to have an error-budget when using spectroscopic redshifts?
        zS = zS_obs
    cosmo = jc.Cosmology(Omega_c=OM, h=H0/100, Omega_k=Ok, w0=w,
                         Omega_b=Ob, wa=wa, sigma8=s8, n_s=n_s)
    # Calculating the theoretical 'r' ratio for given lens/source redshifts and cosmology:
    if cosmo_type in ['FlatLambdaCDM','FlatwCDM']: r_theory = j_r_SL_flat(zL,zS,cosmo)
    else: r_theory = j_r_SL(zL,zS,cosmo)
    # If contaminated, need to know the prior probability each system is a lens (P_tau_0):
    if contaminated:
        P_tau = P_tau.astype('float') #Needs to be a float for dist.Categorical to work
    if photometric and contaminated:
        print('Assuming contaminated, with photometric redshifts')
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
            prob_2 = 0;prob_3 = 0 #This is accounted for by sampling from truncated normal distributions, not a uniform distribution.
            if not no_parent:
                assert False #Haven't checked that this implementation is up to date with the photometric likelihood function.
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
        if True:
            prob = photometric_and_contaminated_likelihood(r_obs,sigma_r_obs,r_theory,
                                            # r_obs[spec_indx],sigma_r_obs[spec_indx],r_theory[spec_indx], NOT Implemented yet
                                            zL_obs,zS_obs,
                                            zL_sigma,zS_sigma,
                                            zL,zS,
                                            mu_zL_g_L=None,mu_zS_g_L = None,
                                            w_zL=None,w_zS=None,
                                            mu_zL_g_L_A=None,mu_zL_g_L_B = None,
                                            sigma_zL_g_L_A=None,sigma_zL_g_L_B=None,
                                            P_tau=P_tau,sigma_r_obs_2=sigma_r_obs_2,
                                            mu_zL_g_NL=None,mu_zS_g_NL=None,
                                            sigma_zL_g_NL=None,sigma_zS_g_NL=None,
                                            sigma_01_g_L=None,sigma_10_g_L=None,
                                            sigma_01_g_NL=None,sigma_10_g_NL=None)
        return prob
    else:
        print('Assuming not contaminated, with spectroscopic redshifts')
        assert False

def j_likelihood_SL(zL_obs,zS_obs,sigma_zL_obs,sigma_zS_obs,r_obs,sigma_r_obs,sigma_r_obs_2=[np.nan],P_tau_0 = [],cosmo_type='',
                    photometric=False,contaminated=False,H0=np.nan,key=None,
                    likelihood_check=False,likelihood_dict = {},cov_redshift=False,early_return=False,
                    batch_bool=True,no_parent=False,
                    trunc_zL=False,trunc_zS=False,P_tau_dist = False,sigma_P_tau = [],lognorm_parent = False,
                    unimodal_beta=True,bimodal_beta=False,true_zL_zS_dep = False,
                    ):
    prior_range_dict = {'OM':(0,1),'Ode':(0,1),'w':(-3,1),'wa':(-3,1),'s_m':(-1,0),'s_c':(0.01,2),'scale_m':(0,6),'scale_c':(0.1,5),
                        'alpha_weights_0':(0,1),'alpha_weights_1':(0,1),
                        'alpha_mu_0':(0,2),'alpha_mu_1':(0,2),'alpha_mu_2':(0,2),
                        'alpha_scale_0':(0.01,5),'alpha_scale_1':(0.01,5),'alpha_scale_2':(0.01,5)}
    step_dict = {'OM':0.05,'Ode':0.05,'w':0.05,'wa':0.05,
                 's_m':0.05,'s_c':0.1,'scale_m':0.5,'scale_c':0.2,
                 'alpha_weights_0':0.05,'alpha_weights_1':0.05,
                 'alpha_mu_0':0.1,'alpha_mu_1':0.1,'alpha_mu_2':0.1,
                 'alpha_scale_0':0.2,'alpha_scale_1':0.2,'alpha_scale_2':0.2,
                 'P_tau':0.005}
    print('Prior dict',prior_range_dict)
    print('Step dict',step_dict)
    N_per_block = 50
    N_steps = 200000
    burnin = 1000
    n_step = 0
    sampling_key=key
    print(f'Sampling Key Start: {sampling_key}')
    N_files = len(glob.glob('/mnt/extraspace/hollowayp/zBEAMS_data/Block_MH_files/Block_MH*'))
    db_out_file = f'/mnt/extraspace/hollowayp/zBEAMS_data/Block_MH_files/Block_MH_0p9_long_{sampling_key}.csv'
    print('Saving file to:',db_out_file)
    N_comp = 3
    acceptance_list = []
    for n_step in tqdm(range(N_steps)):
        acceptance_list = acceptance_list[-20:]
        if n_step<burnin:
            if np.mean(acceptance_list)<0.1 and len(acceptance_list)>=20:
                for k_i in step_dict.keys():
                    if k_i in ['OM','Ode','w','wa','P_tau']:
                        continue
                    else:
                        step_dict[k_i]*=0.9
                print('Increasing step size to ',step_dict)
                acceptance_list = [] #Resetting the acceptance list
        if n_step==0:
            current_dict = {}
            out_of_range_bool=True
            while out_of_range_bool==True:
                out_of_range_bool=False
                for elem in prior_range_dict.keys():
                    #Sample from the prior:
                    current_dict[elem] = float(numpyro.distributions.Uniform(*prior_range_dict[elem]).sample(PRNGKey(sampling_key)))
                    sampling_key+=1
                if (current_dict['alpha_weights_0']+current_dict['alpha_weights_1'])>1:
                    print(f"Initialisation OOD, alpha_weights {current_dict['alpha_weights_0']},{current_dict['alpha_weights_1']}")
                    out_of_range_bool=True
            current_dict['zL'] = jnp.squeeze(numpyro.distributions.TruncatedNormal(loc=jnp.array(zL_obs),scale=jnp.array(sigma_zL_obs),low=0).sample(
                                    PRNGKey(sampling_key)))
            sampling_key+=1
            current_dict['zS'] = jnp.squeeze(numpyro.distributions.TruncatedNormal(loc=jnp.array(zS_obs),scale=jnp.array(sigma_zS_obs),
                                    low=jnp.array(current_dict['zL'])).sample(PRNGKey(sampling_key)))
            sampling_key+=1
            current_dict['simplex_sample'] = jnp.array([current_dict['alpha_weights_0'],
                                                        current_dict['alpha_weights_1'],
                                                        1-(current_dict['alpha_weights_0']+current_dict['alpha_weights_1'])])
            current_dict['alpha_mu_dict'] = {elem:jnp.array(current_dict[f'alpha_mu_{elem}']) for elem in range(N_comp)}
            current_dict['alpha_scale_dict'] = {elem:jnp.array(current_dict[f'alpha_scale_{elem}']) for elem in range(N_comp)}
            current_dict['Ok'] = 1-(current_dict['OM']+current_dict['Ode']) 
            print('THIS NEEDS FIXING AS OTHERWISE THEY ARE NOT RANDOM.')
            current_dict['P_tau'] = P_tau_0 
            for cosmo_param in ['w','wa','OM','Ode','Ok']:
                current_dict[cosmo_param] = float(current_dict[cosmo_param])
            sampling_key+=1
            st = time.time()
            j_likelihood_SL_batch(zL_obs,zS_obs,
                                    sigma_zL_obs,sigma_zS_obs,
                                    r_obs,sigma_r_obs,
                                    sigma_r_obs_2=[sigma_r_obs_2],
                                    P_tau_0 = P_tau_0,cosmo_type=cosmo_type,
                                    photometric=photometric,contaminated=contaminated,H0=H0,key=key,
                                    no_parent=no_parent,
                                    trunc_zL=trunc_zL,trunc_zS=trunc_zS,P_tau_dist = P_tau_dist,
                                    sigma_P_tau = sigma_P_tau,
                                    lognorm_parent = lognorm_parent,
                                    unimodal_beta=unimodal_beta,bimodal_beta=bimodal_beta,true_zL_zS_dep = true_zL_zS_dep,
                                    sampling_dict = current_dict)
            mt=time.time()
            j_likelihood_SL_batch(zL_obs,zS_obs,
                                    sigma_zL_obs,sigma_zS_obs,
                                    r_obs,sigma_r_obs,
                                    sigma_r_obs_2=[sigma_r_obs_2],
                                    P_tau_0 = P_tau_0,cosmo_type=cosmo_type,
                                    photometric=photometric,contaminated=contaminated,H0=H0,key=key,
                                    no_parent=no_parent,
                                    trunc_zL=trunc_zL,trunc_zS=trunc_zS,P_tau_dist = P_tau_dist,
                                    sigma_P_tau = sigma_P_tau,
                                    lognorm_parent = lognorm_parent,
                                    unimodal_beta=unimodal_beta,bimodal_beta=bimodal_beta,true_zL_zS_dep = true_zL_zS_dep,
                                    sampling_dict = current_dict)
            et=time.time()
            print('Uncompiled time',mt-st)
            print('Compiled time',et-mt)
            print('Starting point:',current_dict)
        else:
            t1 = time.time()
            proposed_dict = {}
            out_of_range_bool=True
            while out_of_range_bool:
                out_of_range_bool=False
                for elem in prior_range_dict.keys():
                    out_of_range_single=True
                    while out_of_range_single:
                        out_of_range_single=False
                        # if elem=='Ode':
                            # print('Current',current_dict[elem],'Diff',numpyro.distributions.Normal(loc=current_dict[elem],scale=step_dict[elem]).sample(PRNGKey(sampling_key)))
                        proposed_dict[elem] = current_dict[elem] + numpyro.distributions.Normal(loc=0,scale=step_dict[elem]).sample(PRNGKey(sampling_key))
                        sampling_key+=1
                        if proposed_dict[elem]<prior_range_dict[elem][0] or proposed_dict[elem]>prior_range_dict[elem][1]:
                            out_of_range_single=True
                            print(f'Proposal OOD, {elem}, Proposed: {proposed_dict[elem]}, Bounds: {prior_range_dict[elem]}')
                if (proposed_dict['alpha_weights_0']+proposed_dict['alpha_weights_1'])>1:
                        print(f'Proposal OOD, alpha_weights, Proposed: {proposed_dict["alpha_weights_0"]},{proposed_dict["alpha_weights_1"]}')
                        out_of_range_bool=True
                proposed_dict['simplex_sample'] = jnp.array([proposed_dict['alpha_weights_0'],
                                                            proposed_dict['alpha_weights_1'],
                                                            1-(proposed_dict['alpha_weights_0']+proposed_dict['alpha_weights_1'])])
                proposed_dict['alpha_mu_dict'] = {elem:jnp.array(proposed_dict[f'alpha_mu_{elem}']) for elem in range(N_comp)}
                proposed_dict['alpha_scale_dict'] = {elem:jnp.array(proposed_dict[f'alpha_scale_{elem}']) for elem in range(N_comp)}
            t2 = time.time()
            proposed_dict['zL'] = jnp.squeeze(numpyro.distributions.TruncatedNormal(loc=jnp.array(zL_obs),scale=jnp.array(sigma_zL_obs),low=0).sample(
                                                PRNGKey(sampling_key)))
            sampling_key+=1
            proposed_dict['zS'] = jnp.squeeze(numpyro.distributions.TruncatedNormal(loc=jnp.array(zS_obs),scale=jnp.array(sigma_zS_obs),
                                                low=jnp.array(proposed_dict['zL'])).sample(PRNGKey(sampling_key)))
            sampling_key+=1
            proposed_dict['Ok'] = 1-(proposed_dict['OM']+proposed_dict['Ode'])
            for cosmo_param in ['w','wa','OM','Ode','Ok']:
                proposed_dict[cosmo_param] = float(proposed_dict[cosmo_param])
            block_update_indx = np.random.choice(np.arange(len(zL_obs)),size=N_per_block)
            t3 = time.time()
            out_of_range_bool=True
            while out_of_range_bool:
                out_of_range_bool=False
                #Minimum value of P_tau is 0, highest is 1:
                d_P_tau = numpyro.distributions.TruncatedNormal(loc=jnp.zeros(len(block_update_indx)),scale=step_dict['P_tau'],
                                                                low = -current_dict['P_tau'][block_update_indx],
                                                                high = (1-current_dict['P_tau'][block_update_indx])).sample(PRNGKey(sampling_key))
                sampling_key+=1
                proposed_dict['P_tau'] = current_dict['P_tau']
                proposed_dict['P_tau'] = proposed_dict['P_tau'].at[block_update_indx].set(proposed_dict['P_tau'][block_update_indx]+d_P_tau)
                if (proposed_dict['P_tau']<0).any() or (proposed_dict['P_tau']>1).any():
                    print(f'Proposal OOD, P_tau, Proposed min/max: {np.min(proposed_dict["P_tau"])},{np.max(proposed_dict["P_tau"])}')
                    out_of_range_bool=True
            t4 = time.time()
            proposed_likelihood = j_likelihood_SL_batch(
                                    zL_obs,zS_obs,
                                    sigma_zL_obs,sigma_zS_obs,
                                    r_obs,sigma_r_obs,
                                    sigma_r_obs_2=[sigma_r_obs_2],
                                    P_tau_0 = P_tau_0,cosmo_type=cosmo_type,
                                    photometric=photometric,contaminated=contaminated,H0=H0,key=key,
                                    no_parent=no_parent,
                                    trunc_zL=trunc_zL,trunc_zS=trunc_zS,P_tau_dist = P_tau_dist,
                                    sigma_P_tau = sigma_P_tau,
                                    lognorm_parent = lognorm_parent,
                                    unimodal_beta=unimodal_beta,bimodal_beta=bimodal_beta,true_zL_zS_dep = true_zL_zS_dep,
                                    sampling_dict = proposed_dict)
            t5 = time.time()
            current_likelihood = j_likelihood_SL_batch(
                                    zL_obs,zS_obs,
                                    sigma_zL_obs,sigma_zS_obs,
                                    r_obs,sigma_r_obs,
                                    sigma_r_obs_2=[sigma_r_obs_2],
                                    P_tau_0 = P_tau_0,cosmo_type=cosmo_type,
                                    photometric=photometric,contaminated=contaminated,H0=H0,key=key,
                                    no_parent=no_parent,
                                    trunc_zL=trunc_zL,trunc_zS=trunc_zS,P_tau_dist = P_tau_dist,
                                    sigma_P_tau = sigma_P_tau,
                                    lognorm_parent = lognorm_parent,
                                    unimodal_beta=unimodal_beta,bimodal_beta=bimodal_beta,true_zL_zS_dep = true_zL_zS_dep,
                                    sampling_dict = current_dict)
            t6 = time.time()
            likelihood_ratio = np.exp(jnp.sum(proposed_likelihood) - jnp.sum(current_likelihood))
            random_number = float(numpyro.distributions.Uniform(0,1).sample(PRNGKey(sampling_key)))
            sampling_key+=1
            if likelihood_ratio < 1 and random_number > likelihood_ratio:
                accept = 0
                print('REJECTING proposal')
                acceptance_list.append(accept)
            else:
                accept = 1
                current_likelihood = proposed_likelihood
                current_dict = proposed_dict
                print('ACCEPTING proposal')
                acceptance_list.append(accept)
            t7 = time.time()
            save_dict = current_dict.copy()
            save_dict.pop('zL')
            save_dict.pop('zS')
            save_dict.pop('P_tau')
            save_dict.pop('simplex_sample')
            save_dict.pop('alpha_mu_dict')
            save_dict.pop('alpha_scale_dict')
            save_dict['likelihood_ratio']=likelihood_ratio
            save_dict['current_likelihood']=jnp.sum(current_likelihood)
            save_dict['accept']=accept
            # print('saving',save_dict)
            if n_step<=1:
                df = pd.DataFrame(save_dict,index=[n_step])
                # print('df',df)
                # print('df_string',df.to_string())
                with open(db_out_file,'w') as f:
                    output_str = " ".join(df.columns) + '\n' + " ".join(df.to_string(header=False).split()) +'\n'
                    f.write(output_str)
            else:
                df = pd.DataFrame(save_dict,index=[n_step])
                with open(db_out_file,'a') as f:
                    output_str = " ".join(df.to_string(header=False).split()) +'\n'
                    f.write(output_str)
            t8 = time.time()
            print('Time taken:',t8-t1,'t1-t2:',t2-t1,'t3-t2:',t3-t2,'t4-t3:',t4-t3,'t5-t4:',t5-t4,'t6-t5:',t6-t5,'t7-t6:',t7-t6,'t8-t7:',t8-t7)
    # gc.collect()
    # total_prob-=jnp.sqrt(N_dim) #Rescaling the likelihood to help convergence.
    # L = numpyro.factor("Likelihood",total_prob)

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
                'H0':H0,'no_parent':no_parent,'trunc_zL':trunc_zL,'trunc_zS':trunc_zS,
                'P_tau_dist':P_tau_dist,'sigma_P_tau':sigma_P_tau,'lognorm_parent':lognorm_parent,
                'unimodal_beta':unimodal_beta,'bimodal_beta':bimodal_beta,'true_zL_zS_dep':true_zL_zS_dep,
                'key':key_int}
    print(f'Model args: {model_args}')
    j_likelihood_SL(**model_args)

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

