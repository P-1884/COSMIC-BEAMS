# from astropy.cosmology import LambdaCDM,FlatLambdaCDM,wCDM,FlatwCDM,w0waCDM
# from numpyro import distributions as dist, infer
# from numpyro.infer import MCMC, NUTS, HMC
# from jax import random,grad, jit
import matplotlib.pyplot as pl
# import jax.numpy as jnp
# import jax_cosmo as jc
from tqdm import tqdm
# import scipy.sparse
# import arviz as az
import pandas as pd
import numpy as np
# import numpyro
#import emcee
import sys
#import jax
# from mcmcfunctions_SL_JAX import j_likelihood_SL,run_MCMC
#az.style.use('arviz-doc')
#sys.path.append('/Users/hollowayp/zBEAMS')
from marginalisation_test import marginalisation_test,marginalisation_test_no_parent,r_func
from scipy.integrate import quad
import multiprocessing as mp
from scipy.stats import norm
np.random.seed(1)

def posterior_i_func(input_tuple):
    O_list = np.linspace(0,1,1000)
    def integrand(z_true,z_obs,r_obs,sigma_z_obs,sigma_r_obs,O_int):
        return norm(z_obs,sigma_z_obs).pdf(z_true)*norm(r_obs,sigma_r_obs).pdf(r_func(z_true,O_int))
    z_obs_i = input_tuple[0]
    r_obs_i = input_tuple[1]
    sigma_z_obs_i = input_tuple[2]
    sigma_r_obs_i = input_tuple[3]
    z_min = input_tuple[4]
    z_max = input_tuple[5]
    print(f'z integration from {z_min} to {z_max}')
    return np.array([quad(integrand,z_min,z_max,args=(z_obs_i,
                                        r_obs_i,
                                        sigma_z_obs_i,
                                        sigma_r_obs_i,elem))[0] for elem in O_list])

O_test = 0.3
N_obs = int(sys.argv[1])
sigma_z_obs = float(sys.argv[2])
try:
    emcee_or_analytic = sys.argv[3]
except:
    emcee_or_analytic='analytic'
try:
    N_steps = int(sys.argv[4])
except:
    N_steps = 5000

sigma_r_obs = 5
zL_true_test = 20

def generate_obs_dict(sigma_z_obs,sigma_r_obs,N_obs):
    np.random.seed(1)
    obs_dict = {
        'z':norm(zL_true_test,sigma_z_obs).rvs(size=N_obs),
        'r':norm(r_func(zL_true_test,O_test),sigma_r_obs).rvs(size=N_obs)}
    return obs_dict

obs_dict = generate_obs_dict(sigma_z_obs,sigma_r_obs,N_obs)

print(f'Input arguments: {N_obs}, {sigma_z_obs},{emcee_or_analytic},'+str(N_steps)*(emcee_or_analytic=='emcee'))

if emcee_or_analytic=='analytic':
    input_tuple = [(obs_dict['z'][obs_i],obs_dict['r'][obs_i],sigma_z_obs,sigma_r_obs,
                    np.min(obs_dict['z'])-4*np.max(sigma_z_obs),
                    np.max(obs_dict['z'])+4*np.max(sigma_z_obs))
                    for obs_i in range(N_obs)]
    with mp.Pool() as Pool:
        a = list(tqdm(Pool.imap(posterior_i_func, input_tuple), total=N_obs))
    P_theta_dict = {elem:np.array(a)[elem,:] for elem in np.arange(N_obs)}
    db_test = pd.DataFrame(P_theta_dict)
else:
    s_test,N_walkers = marginalisation_test_no_parent(obs_dict['z'],obs_dict['r'],
                              sigma_z_obs,sigma_r_obs,N_steps=N_steps,verbose=False,
                              sampler_kwargs={'pool':mp.Pool()})
    db_test = pd.DataFrame({chain_i:s_test.get_chain()[:,chain_i,0] for chain_i in range(N_obs)})
    db_test_z = {}
    for chain_i in range(2*N_obs+1):
        for obs_i in range(1,N_obs+1):
            db_test_z[f'{chain_i}_{obs_i}'] = s_test.get_chain()[:,chain_i,obs_i]
    db_test_z = pd.DataFrame(db_test_z)

file_out = f'/mnt/zfsusers/hollowayp/zBEAMS/chains/MCMC_tests/MCMC_toy_{emcee_or_analytic}_{N_obs}_{sigma_z_obs}.csv'
print(f'Saving file to {file_out}')
db_test.to_csv(file_out,index=False)

if emcee_or_analytic=='emcee':
    file_out_z = f'/mnt/zfsusers/hollowayp/zBEAMS/chains/MCMC_tests/MCMC_toy_{emcee_or_analytic}_z_{N_obs}_{sigma_z_obs}.csv'
    print(f'Saving redshift chains file to {file_out_z}')
    #Applying thinning to the redshift chains, to save memory:
    db_test_z=db_test_z.loc[np.arange(0,N_steps,100).astype('int')]
    db_test_z.to_csv(file_out_z)
