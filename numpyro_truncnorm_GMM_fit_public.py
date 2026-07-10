from numpyro import distributions as dist, infer
from numpyro.infer import MCMC, NUTS
from jax import random,grad, jit
import matplotlib.pyplot as pl
import jax.numpy as jnp
import pandas as pd
import numpy as np
import numpyro

def numpyro_truncnorm_GMM_fit(data, #Array to fit a truncated gaussian mixture model (GMM) to.
                              N_comp = 1, #Number of components in the GMM.
                              num_warmup=1000, #Number of warmup steps for the MCMC sampler.
                              num_samples=1000, #Number of samples to draw from the MCMC sampler.
                              return_all_samples=False, #If True, returns all MCMC samples. If False, returns only the last sample of a single chain.
                              ):
    def model(data):
        GMM_mu_dict = {elem:numpyro.sample(f'alpha_mu_{elem}',dist.Uniform(0,20),sample_shape=(1,)) for elem in range(N_comp)}
        GMM_scale_dict = {elem:numpyro.sample(f'alpha_scale_{elem}',dist.LogUniform(0.001,5),sample_shape=(1,)) for elem in range(N_comp)}
        GMM_weights = numpyro.sample('alpha_weights',
                                        dist.Dirichlet(concentration=jnp.array([1.0]*N_comp)))
        prob = dist.Mixture(dist.Categorical(GMM_weights.T),
                            [dist.TruncatedNormal(loc=GMM_mu_dict[elem],
                                               scale=GMM_scale_dict[elem],low=0) for elem in range(N_comp)],
                             ).log_prob(data)
        L = numpyro.factor('Likelihood',prob)
    kernel = NUTS(model)
    mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples,num_chains=1)
    mcmc.run(random.PRNGKey(0), data)
    mcmc_samples = mcmc.get_samples()
    if return_all_samples: return mcmc_samples
    else: print('NB: Just returning last sample of a single chain')
    list_of_mu = [float((mcmc_samples[f'alpha_mu_{elem}']).flatten()[-1]) for elem in range(N_comp)]
    list_of_sigma = [float((mcmc_samples[f'alpha_scale_{elem}']).flatten()[-1]) for elem in range(N_comp)]
    list_of_weights = [float((mcmc_samples['alpha_weights'][:,elem]).flatten()[-1]) for elem in range(N_comp)]
    print('list_of_mu',list_of_mu,'list_of_sigma',list_of_sigma,'list_of_weights',list_of_weights)
    return {'list_of_mu':list_of_mu,'list_of_sigma':list_of_sigma,'list_of_weights':list_of_weights}
