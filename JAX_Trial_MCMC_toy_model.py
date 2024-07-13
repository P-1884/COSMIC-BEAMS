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
import emcee
import sys
import jax
# from mcmcfunctions_SL_JAX import j_likelihood_SL,run_MCMC
#az.style.use('arviz-doc')
#sys.path.append('/Users/hollowayp/zBEAMS')
from marginalisation_test import marginalisation_test,r_func
from scipy.stats import norm
np.random.seed(1)

sigma_dict = {}
N_obs = int(sys.argv[1])
sigma_dict['z'] = float(sys.argv[2])
sigma_dict['r'] = 5
print(f'Input arguments: {N_obs}, {sigma_dict["z"]}',sys.argv)
method = sys.argv[3]
assert method in ['jax','emcee']
if method =='emcee':
    try:
        N_steps = int(sys.argv[4])
    except: N_steps=5000
else:
    from marginalisation_test import j_marginalisation,run_MCMC_marginalisation
    import jax.numpy as jnp 
    import numpyro
    from numpyro import distributions as dist
    from jax.random import PRNGKey
    try: N_warmup = int(sys.argv[4])
    except: N_warmup = 1000
    try: N_samples = int(sys.argv[5])
    except: N_samples = 1000
    try: 
        mcmc_type = sys.argv[6]
        print('MCMC type',mcmc_type)
        if mcmc_type=='batch': batch=True;barker=False
        elif mcmc_type=='barker': batch=False;barker=True
        else: batch=False;barker=False
        print('Batch bool:',batch,'Barker bool:',barker)
    except: 
        batch=False
        barker=False

O_test = 0.3

def generate_obs_dict():
    np.random.seed(1)
    obs_dict = {
    'z':norm(zL_true_test,sigma_dict['z']).rvs(),
    'r':norm(r_func(zL_true_test,O_test),sigma_dict['r']).rvs()}
    return obs_dict

zL_true_test = np.array(norm(20,5).rvs(size=N_obs))
obs_dict = generate_obs_dict()

if method=='emcee':
    s_test,N_walkers = marginalisation_test(obs_dict['z'],obs_dict['r'],
                              sigma_dict['z'],sigma_dict['r'],N_steps=N_steps)
    chain_dict = {}
    for n_chain in range(s_test.get_chain().shape[1]):
        for n_z in np.arange(3,s_test.get_chain().shape[2]):
            chain_dict[f'z_{n_z}_{n_chain}'] = s_test.get_chain()[:,n_chain,n_z]
        chain_dict[f'theta_{n_chain}'] = s_test.get_chain()[:,n_chain,0]
        chain_dict[f'mu_z_{n_chain}'] = s_test.get_chain()[:,n_chain,1]
        chain_dict[f'sigma_z_{n_chain}'] = s_test.get_chain()[:,n_chain,2]
    db_test = pd.DataFrame(chain_dict)[[f'theta_{ii}' for ii in range(N_walkers)]]
else:
    z_obs = jnp.array(obs_dict['z'])
    r_obs = jnp.array(obs_dict['r'])
    N_chains = 50
    jax_sampler = run_MCMC_marginalisation(z_obs,r_obs,sigma_dict['z'],sigma_dict['r'],
                            num_chains=N_chains,
                            num_warmup = N_warmup,
                            num_samples = N_samples,
                            target_accept_prob=0.99,
                            batch=batch,
                            barker=barker)
    db_test = pd.DataFrame({f'theta_{chain_i}':jax_sampler.get_samples(True)['theta'][chain_i,:,0] 
                            for chain_i in range(N_chains)})

file_out = f'/mnt/zfsusers/hollowayp/zBEAMS/chains/MCMC_tests/MCMC_toy_'+'JAX_'*(method=='jax')+\
           f'{N_obs}_{sigma_dict["z"]}.csv'
print(f'Saving file to {file_out}')
db_test.to_csv(file_out,index=False)