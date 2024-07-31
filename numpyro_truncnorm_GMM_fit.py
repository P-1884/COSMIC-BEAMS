from numpyro import distributions as dist, infer
from numpyro.infer import MCMC, NUTS
from jax import random,grad, jit
import jax.numpy as jnp
import numpyro

def numpyro_truncnorm_GMM_fit(data,N_comp = 1,num_warmup=1000,num_samples=1000,return_all_samples=False,inf_if_weights_unordered=False):
    def model(data):
        alpha_mu_dict = {elem:numpyro.sample(f'alpha_mu_{elem}',dist.Uniform(0,5),sample_shape=(1,)) for elem in range(N_comp)}
        alpha_scale_dict = {elem:numpyro.sample(f'alpha_scale_{elem}',dist.LogUniform(0.01,5),sample_shape=(1,)) for elem in range(N_comp)}
        simplex_sample = numpyro.sample('alpha_weights',
                                        dist.Dirichlet(concentration=jnp.array([1.0]*N_comp)))
        # jax.debug.print('shape {a}',a=simplex_sample.shape)
        # alpha_w0 = numpyro.sample('alpha_w0',dist.Uniform(0,1),sample_shape=(1,))
        # simplex_sample = numpyro.deterministic('alpha_weights',jnp.array([alpha_w0,1-alpha_w0]))
        # jax.debug.print('a {a}',a=simplex_sample)
        prob = dist.Mixture(dist.Categorical(simplex_sample.T),
                            [dist.TruncatedNormal(loc=alpha_mu_dict[elem],
                                               scale=alpha_scale_dict[elem],low=0) for elem in range(N_comp)],
                             ).log_prob(data)
        # prob_dict = {elem:dist.TruncatedNormal(loc=alpha_mu_dict[elem],
                                            #    scale=alpha_scale_dict[elem],low=0).log_prob(data) for elem in range(N_comp)}
        # prob = jnp.array([jnp.log(simplex_sample[elem])+jnp.sum(prob_dict[elem]) for elem in range(N_comp)])
        # prob = jnp.sum(prob,axis=0)
        # jax.debug.print('prob {prob}',prob=prob.shape)
        if inf_if_weights_unordered:
            prob = jnp.where(simplex_sample[0]>simplex_sample[1],-jnp.inf,prob)
            prob = jnp.where(simplex_sample[1]>simplex_sample[2],-jnp.inf,prob)
        L = numpyro.factor('Likelihood',prob)
    kernel = NUTS(model)
    mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples,num_chains=1)
    mcmc.run(random.PRNGKey(0), data)
    mcmc_samples = mcmc.get_samples()
    if return_all_samples: return mcmc_samples
    print('NB: JUST returning last sample of a single chain - am not taking the mean or median or anything')
    # print(mcmc_samples['alpha_mu_0'].shape)
    list_of_mu = [float((mcmc_samples[f'alpha_mu_{elem}']).flatten()[-1]) for elem in range(N_comp)]
    list_of_sigma = [float((mcmc_samples[f'alpha_scale_{elem}']).flatten()[-1]) for elem in range(N_comp)]
    list_of_weights = [float((mcmc_samples['alpha_weights'][:,elem]).flatten()[-1]) for elem in range(N_comp)]
    # pl.plot(mcmc_samples['alpha_weights'])
    # pl.show()
    return {'list_of_mu':list_of_mu,'list_of_sigma':list_of_sigma,'list_of_weights':list_of_weights}