import numpy as np
import jax
import numpyro
from numpyro.infer import MCMC, NUTS
import jax.numpy as jnp
import numpyro.distributions as dist

import wandb
import initialise_wandb
import os
import gc
os.environ['WANDB_API_KEY'] #Checking key exists
os.environ['XLA_PYTHON_CLIENT_ALLOCATOR']='platform'
N_data = 100000
N_warmup = 100000
N_samples = 100000
wandb.init(
    # set the wandb project where this run will be logged
    project="NUTS Memory Profiling",
    # track hyperparameters and run metadata
    config={
    "N_data": N_data,
    "N_warmup": N_warmup,
    "N_samples": N_samples,
    'Comment':'Platform allocator, 10 runs,no gc'
    }
)

# Define the model
def gaussian_model(args):
    data,mu,sigma=args
    return jnp.sum(dist.Normal(mu,sigma).log_prob(data))

def gaussian_model_full(data):
    # L=0
    mu = numpyro.sample('mu', dist.Normal(0, 10))
    sigma = numpyro.sample('sigma', dist.HalfNormal(10))
    L = numpyro.factor('L',dist.Normal(mu,sigma).log_prob(data))

# Generate some synthetic data
np.random.seed(0)
true_mu = 1.0
true_sigma = 2.0
data = np.random.normal(true_mu, true_sigma, size=N_data)

# Run the MCMC sampler
nuts_kernel = NUTS(gaussian_model_full)
mcmc = MCMC(nuts_kernel, num_warmup=N_warmup, num_samples=N_samples)
mcmc.warmup(jax.random.PRNGKey(0), data)
# mcmc.post_warmup_state = mcmc.last_state
# mcmc.run(mcmc.post_warmup_state.rng_key,data)
# mcmc.print_summary()

for i in range(10):
    mcmc.run(mcmc.post_warmup_state.rng_key,data)
    mcmc.post_warmup_state = mcmc.last_state
    # gc.collect()

mcmc.print_summary()
