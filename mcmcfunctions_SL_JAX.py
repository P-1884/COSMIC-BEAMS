from astropy.cosmology import LambdaCDM,FlatLambdaCDM,wCDM,FlatwCDM,w0waCDM
from numpyro import distributions as dist, infer
from numpyro.infer import MCMC, NUTS, HMC
from jax import random,grad, jit
import matplotlib.pyplot as pl
import jax.numpy as jnp
import jax_cosmo as jc
from tqdm import tqdm
import scipy.sparse
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
def MVN_samp(loc_0,loc_1,sig_0,sig_1,x0,x1):
        return dist.MultivariateNormal(loc=jnp.array([loc_0,loc_1]),
                        covariance_matrix=jnp.array([[sig_0,0],[0,sig_1]])).log_prob(jnp.array([x0,x1]).T) 

@jit
def likelihood_PC(P_tau,prob_1a,prob_1b,prob_2,prob_3,prob_4a,prob_4b):
        return (jnp.log(P_tau*jnp.exp(prob_1a)*jnp.exp(prob_4a)+(1-P_tau)*jnp.exp(prob_1b)*jnp.exp(prob_4b))+prob_2+prob_3)


def j_likelihood_SL(zL_obs,zS_obs,sigma_zL_obs,sigma_zS_obs,r_obs,sigma_r_obs,sigma_r_obs_2=[np.nan],P_tau = [],cosmo_type='',
                    photometric=False,contaminated=False,H0=np.nan,key=None,likelihood_check=False,likelihood_dict = {}):
    jax.random.PRNGKey(0)
    s8 = 0.8;n_s = 0.96;Ob=0; #Putting all the matter in dark-matter (doesn't make a difference)
    if not likelihood_check:
        if photometric:
            print('Assuming photometric redshifts')
            zL_sigma = sigma_zL_obs;zS_sigma = sigma_zS_obs
            #Have REMOVED lower-bound of zS being higher than zL - perhaps need to reinstate?
            zL_unscaled = numpyro.sample('zL_unscaled',dist.TruncatedNormal(0,scale=jnp.ones(zL_obs.shape),low=0),sample_shape=(1,),
                                    rng_key=key).flatten()
            zS_unscaled = numpyro.sample('zS_unscaled',dist.TruncatedNormal(0,scale=jnp.ones(zS_obs.shape),low=0),sample_shape=(1,),
                                        rng_key=key).flatten()
            zL = numpyro.deterministic('zL',jnp.array(zL_obs)+zL_unscaled*jnp.array(zL_sigma))
            zS = numpyro.deterministic('zS',jnp.array(zS_obs)+zS_unscaled*jnp.array(zS_sigma))
            #zL = numpyro.sample('zL',dist.TruncatedNormal(jnp.array(zL_obs),zL_sigma,low=0),sample_shape=(1,),rng_key=key).flatten()
            #zS = numpyro.sample('zS',dist.TruncatedNormal(jnp.array(zS_obs),zS_sigma,low=0),sample_shape=(1,),rng_key=key).flatten()
            mu_zL_g_L = jnp.squeeze(numpyro.sample("mu_zL_g_L", dist.Uniform(0,0.5),sample_shape=(1,),rng_key=key))
            mu_zS_g_L = jnp.squeeze(numpyro.sample("mu_zS_g_L", dist.Uniform(0.5,1),sample_shape=(1,),rng_key=key))
            sigma_zL_g_L = jnp.squeeze(numpyro.sample("sigma_zL_g_L", dist.Uniform(0.1,2),sample_shape=(1,),rng_key=key))
            sigma_zS_g_L = jnp.squeeze(numpyro.sample("sigma_zS_g_L", dist.Uniform(0.1,2),sample_shape=(1,),rng_key=key))
        else:
            print('Assuming spectroscopic redshifts')
            zL = zL_obs #Think still need to have an error-budget when using spectroscopic redshifts?
            zS = zS_obs
        OM = jnp.squeeze(numpyro.sample("OM", dist.Uniform(0,1),sample_shape=(1,),rng_key=key))
        if cosmo_type in ['FlatLambdaCDM','FlatwCDM']:
            #Don't care about Ode, as it isn't an argument for the cosmology (OM and Ok are instead)
            print('Assuming a flat universe')
            #Somehow this bugs. It is a problem with the value 0, and not with Ok being fixed (0.01 doesn't bug):
            Ok = numpyro.deterministic('Ok',0.0)
        else:
            print('Assuming the universe may have curvature')
            print('NOTE: Need to think about my priors more here - if there is no information, the Om and Ode priors are uniform, so my\n '+\
                'Ok prior would be triangular (1-(Ode+Om)), centered on 0. **This is very problematic**. Perhaps need to sample from a uniform\n '+\
                '3D distribution with some planar cuts to make Ode+Ok+Om=1?')
            Ode = jnp.squeeze(numpyro.sample("Ode", dist.Uniform(0,1),sample_shape=(1,),rng_key=key))
            Ok = jnp.array(1-(OM+Ode))
        if cosmo_type in ['LambdaCDM','FlatLambdaCDM']:
            print('Assuming universe has a cosmological constant')
            w = numpyro.deterministic('w',-1.0)
            wa = numpyro.deterministic('wa',0.0)
        else:
            print('Assuming non-trivial dark energy equation of state')
            w = jnp.squeeze(numpyro.sample("w", dist.Uniform(-6,4),sample_shape=(1,),rng_key=key)) #Physicality constraints
            wa = jnp.squeeze(numpyro.sample("wa", dist.Uniform(-3,1),sample_shape=(1,),rng_key=key)) #Matching Tian's constraints for now
        '''
        If this code turns out to be quite slow, the previous likelihood function used interpolation.
    '''
    if likelihood_check:
            OM = likelihood_dict['OM'];Ok = likelihood_dict['Ok']
            w = likelihood_dict['w'];wa=likelihood_dict['wa'];
            r_obs =likelihood_dict['r_obs'];sigma_r_obs = likelihood_dict['sigma_r_obs'];
            zL = likelihood_dict['zL'];zS = likelihood_dict['zS']
            zL_obs = likelihood_dict['zL_obs'];zS_obs = likelihood_dict['zS_obs']
            zL_sigma = likelihood_dict['zL_sigma'];zS_sigma = likelihood_dict['zS_sigma']
            mu_zL_g_L,mu_zS_g_L = likelihood_dict['mu_zL_g_L'],likelihood_dict['mu_zS_g_L']
            sigma_zL_g_L,sigma_zS_g_L = likelihood_dict['sigma_zL_g_L'],likelihood_dict['sigma_zS_g_L']
    cosmo = jc.Cosmology(Omega_c=OM,h=H0/100, Omega_k=Ok,w0=w,
                         Omega_b=Ob,wa=wa,sigma8=s8,n_s=n_s)
    if cosmo_type in ['FlatLambdaCDM','FlatwCDM']: r_theory = j_r_SL_flat(zL,zS,cosmo)
    else: r_theory = j_r_SL(zL,zS,cosmo)
    if contaminated and not photometric:
        P_tau = P_tau.astype('float') #Needs to be a float for dist.Categorical to work
        print('Assuming contaminated, with spectroscopic redshifts')
        assert not np.isnan(sigma_r_obs_2)
        print('NOTE: Need to come up with a test function (e.g. known likelihood) to see what this is actually doing, rather than just\n'+\
            ' assuming that because it gives the right answer it must be correct.')
        prob_1 = dist.Mixture(dist.Categorical(jnp.array([P_tau, 1-P_tau]).T),
                            [dist.Normal(r_theory, sigma_r_obs),
                             dist.Normal(r_theory, sigma_r_obs_2)]).log_prob(r_obs)
        L = numpyro.factor("Likelihood",prob_1)
    elif photometric and not contaminated:
        print('Assuming not contaminated, with photometric redshifts')
        print('NOTE: Need to come up with a test function (e.g. known likelihood) to see what this is actually doing, rather than just\n'+\
            ' assuming that because it gives the right answer it must be correct.')
        #numpyro.factor allows you to add more components to the likelihood, or just do dist.Normal()*dist.Normal() if product of two
        #normal distributions.
        #NUMPYRO.SAMPLE DOESN'T NEED OBS KWARG? 
        # print('Cosmo Input: ',cosmo)
        # print(
        #     f'Inputs: r_theory: {r_theory}, r_obs: {r_obs}, sigma_r_obs:{sigma_r_obs},\n'+\
        #     f'zL: {zL}, zS: {zS},\nzL_obs: {zL_obs}, zS_obs: {zS_obs},\nsigma_zL: {zL_sigma}, sigma_zS: {zS_sigma},\n'+\
        #     f'mu_zL_g_L: {mu_zL_g_L}, mu_zS_g_L: {mu_zS_g_L},\nsigma_zL_g_L: {sigma_zL_g_L}, sigma_zS_g_L: {sigma_zS_g_L}')
        prob_1 = dist.Normal(r_theory, sigma_r_obs).log_prob(r_obs)
        prob_2 = dist.Normal(zL, zL_sigma).log_prob(zL_obs)
        prob_3 = dist.Normal(zS, zS_sigma).log_prob(zS_obs)
        #Assuming diagonal covariance matrix for now:
        prob_4 = dist.MultivariateNormal(loc=jnp.array([mu_zL_g_L,mu_zS_g_L]),
                           covariance_matrix=jnp.array([[sigma_zL_g_L,0],
                                                        [0,sigma_zS_g_L]])).log_prob(jnp.array([zL,zS]).T)
        #Assuming a uniform prior on zL and zS => This doesn't actually do anything as it breaks if it returns -np.inf outside the 
        #prior so I think this will now just always return 0 regardless of the input arguments.
        # prob_4 = dist.Uniform(low=0,high=2).log_prob(zL)+\
        #          dist.Uniform(low=0,high=2).log_prob(zS)
        L = numpyro.factor("Likelihood",prob_1+prob_2+prob_3+prob_4)
        print('Likelihood Out:',L,prob_1+prob_2+prob_3+prob_4)
        if likelihood_check: return prob_1+prob_2+prob_3+prob_4
    elif photometric and contaminated:
        P_tau = P_tau.astype('float') #Needs to be a float for dist.Categorical to work
        print('Assuming contaminated, with photometric redshifts')
        print('NOTE: Need to come up with a test function (e.g. known likelihood) to see what this is actually doing, rather than just\n'+\
            ' assuming that because it gives the right answer it must be correct.')
        mu_zL_g_NL = jnp.squeeze(numpyro.sample("mu_zL_g_NL", dist.Uniform(0,5),sample_shape=(1,),rng_key=key))
        mu_zS_g_NL = jnp.squeeze(numpyro.sample("mu_zS_g_NL", dist.Uniform(0,5),sample_shape=(1,),rng_key=key))
        sigma_zL_g_NL = jnp.squeeze(numpyro.sample("sigma_zL_g_NL", dist.Uniform(0.1,5),sample_shape=(1,),rng_key=key))
        sigma_zS_g_NL = jnp.squeeze(numpyro.sample("sigma_zS_g_NL", dist.Uniform(0.1,5),sample_shape=(1,),rng_key=key))
        # dist.Mixture(dist.Categorical(jnp.array([P_tau, 1-P_tau]).T),
                   #         [dist.Normal(r_theory, sigma_r_obs),
                    #         dist.Normal(r_theory, sigma_r_obs_2)]).log_prob(r_obs)
        prob_1a = dist.Normal(r_theory, sigma_r_obs).log_prob(r_obs)
        prob_1b = dist.Normal(r_theory, sigma_r_obs_2).log_prob(r_obs)
        prob_2 = dist.Normal(zL, zL_sigma).log_prob(zL_obs)
        prob_3 = dist.Normal(zS, zS_sigma).log_prob(zS_obs)
        #Assuming diagonal covariance matrix for now:
        prob_4a = (MVN_samp)(mu_zL_g_L,mu_zS_g_L,sigma_zL_g_L,sigma_zS_g_L,zL,zS)
                        #dist.MultivariateNormal(loc=jnp.array([mu_zL_g_L,mu_zS_g_L]),
                        #                        covariance_matrix=jnp.array([[sigma_zL_g_L,0],
                        #                                                    [0,sigma_zS_g_L]])).log_prob(jnp.array([zL,zS]).T)
        prob_4b = (MVN_samp)(mu_zL_g_NL,mu_zS_g_NL,sigma_zL_g_NL,sigma_zS_g_NL,zL,zS)
                        #dist.MultivariateNormal(loc=jnp.array([mu_zL_g_NL,mu_zS_g_NL]),
                        #                       covariance_matrix=jnp.array([[sigma_zL_g_NL,0],
                        #                                                    [0,sigma_zS_g_NL]])).log_prob(jnp.array([zL,zS]).T)
        '''
        Seems to be a problem with very small numbers - can cope if I increase the precision but still with only very small numbers of 
        systems => Problem fixed by having P_tau!=1.0 (even 0.9 fixed it).
        '''
        # print('1a',(prob_1a))
        # print('1b',(prob_1b))
        # print('2',prob_2)
        # print('3',prob_3)
        # print('4a',(prob_4a))
        # print('4b',(prob_4b))
        # print('P_tau',P_tau)
        # print('Exp_a',jnp.exp(prob_1a+prob_4a))
        # print('Exp_b',jnp.exp(prob_1b+prob_4b))
        #print('TYPE',type(prob_1a[0]),type(prob_1b[0]),type(prob_4a[0]),type(prob_4b[0]))
        #print('Sum',P_tau*jnp.exp(prob_1a+prob_4a)+(1-P_tau)*jnp.exp(prob_1b+prob_4b))
        #'log_prob' finds the natural logarithm (not log10), hence these are natural-logged:
        L_1 =  likelihood_PC(P_tau,prob_1a,prob_1b,prob_2,prob_3,prob_4a,prob_4b)
                #jnp.log(P_tau*jnp.exp(prob_1a)*jnp.exp(prob_4a)+(1-P_tau)*jnp.exp(prob_1b)*jnp.exp(prob_4b))+prob_2+prob_3
        #print('L1',L_1)
        L = numpyro.factor("Likelihood",L_1)
        #print("L",L)
    else:
        print('Assuming not contaminated, with spectroscopic redshifts')
        assert not photometric and not contaminated
        #assert (sigma_r_obs>0).all()
        numpyro.sample("r", dist.Normal(r_theory, sigma_r_obs), obs=r_obs)


def run_MCMC(photometric,contaminated,cosmo_type,
            zL_obs,zS_obs,sigma_zL_obs,sigma_zS_obs,
            r_obs,sigma_r_obs,sigma_r_obs_2,P_tau,
            num_warmup = 200,num_samples=1000,num_chains=2,H0=np.nan,target_accept_prob=0.8):
    print(f'Target Accept Prob: {target_accept_prob}')
    st = time.time()
    j_likelihood_SL(zL_obs=zL_obs,zS_obs=zS_obs,sigma_zL_obs=sigma_zL_obs,sigma_zS_obs=sigma_zS_obs,r_obs=r_obs,sigma_r_obs=sigma_r_obs,sigma_r_obs_2=sigma_r_obs_2,P_tau=P_tau,cosmo_type=cosmo_type,photometric=photometric,contaminated=contaminated,H0=H0,key=jax.random.PRNGKey(0))
    mt=time.time()
    j_likelihood_SL(zL_obs=zL_obs,zS_obs=zS_obs,sigma_zL_obs=sigma_zL_obs,sigma_zS_obs=sigma_zS_obs,r_obs=r_obs,sigma_r_obs=sigma_r_obs,sigma_r_obs_2=sigma_r_obs_2,P_tau=P_tau,cosmo_type=cosmo_type,photometric=photometric,contaminated=contaminated,H0=H0,key=jax.random.PRNGKey(0))
    et=time.time()
    print('Uncompiled time',mt-st)
    print('Compiled time',et-mt)
    sampler_0 = infer.MCMC(
        infer.NUTS(model = j_likelihood_SL,
                   target_accept_prob = target_accept_prob),
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        progress_bar=True)
    sampler_0.run(jax.random.PRNGKey(0),
                zL_obs=zL_obs,
                zS_obs=zS_obs,
                sigma_zL_obs=sigma_zL_obs,
                sigma_zS_obs=sigma_zS_obs,
                r_obs=r_obs,
                sigma_r_obs=sigma_r_obs,
                sigma_r_obs_2=sigma_r_obs_2,
                P_tau=P_tau,
                cosmo_type=cosmo_type,
                photometric=photometric,
                contaminated=contaminated,
                H0=H0)
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
