from scipy.stats import lognorm,truncnorm
from JAX_samples_to_dict import JAX_samples_to_dict
from numpyro import distributions as dist,infer
from numpyro.infer import MCMC,NUTS,HMC,HMCECS,init_to_value,init_to_uniform
from jax import random,grad, jit
import matplotlib.pyplot as pl
import jax.numpy as jnp
from tqdm import tqdm
import scipy.sparse
import pandas as pd
#import arviz as az
import numpy as np
import numpyro
# import corner
import sys
import jax
import time
from jax.scipy.stats import truncnorm as jax_truncnorm
from Beta_Distribution_Class import beta_class
from LogNormal_Distribution_Class import jax_lognormal 
jax.config.update("jax_enable_x64", True)
numpyro.enable_x64(True)
seed = int(sys.argv[1])
print(f'Seed: {seed}')
# from numpyro.contrib.nested_sampling import NestedSampler
# from jaxns import DefaultNestedSampler as NestedSampler
import glob
from scipy.interpolate import interp1d
class sample_from_jax_truncnorm_cdf:
    def __init__(self,loc,scale,N,seed):
        np.random.seed(seed)
        self.loc = loc
        self.scale = scale
        self.N = N
        self.a = -loc/scale
        self.b = np.inf
        self.rand_val = np.random.uniform(size=self.N)
    def sample_from_CDF(self):
        X_interp = np.linspace(0,50,100000)
        CDF_interp = jax_truncnorm.cdf(x=X_interp,a=self.a,b=self.b,loc=self.loc,scale=self.scale)
        return interp1d(CDF_interp,X_interp)(self.rand_val) 

class generate_lognorm_data:
    def __init__(self,loc,scale,s,N,sigma_r_obs):
        self.loc = loc
        self.scale = scale
        self.s = s
        self.N = N
        self.sigma_r_obs = sigma_r_obs
    def generate_r_true(self):
        np.random.seed(seed)
        self.r_true = lognorm(loc=self.loc,scale=self.scale,s=self.s).rvs(self.N)
    def generate_r_obs(self):
        self.generate_r_true()
        self.r_true.sort()
        self.sigma_r_obs.sort()
        np.random.seed(seed+1)
        self.r_obs = truncnorm(a=-self.r_true/self.sigma_r_obs,b=np.inf,loc=self.r_true,scale=self.sigma_r_obs).rvs(self.N)
        # self.r_obs = jnp.array([sample_from_jax_truncnorm_cdf(loc=self.r_true[i],scale=self.sigma_r_obs,N=1,seed=i).sample_from_CDF() for i in tqdm(range(self.N))])
        return self.r_obs,self.sigma_r_obs,self.r_true

db_10k_FPs = pd.read_csv('./databases/10k_FPs.csv')
N_sys = int(sys.argv[2])
r_obs,sigma_r_obs,r_true = generate_lognorm_data(s = 0.3,loc = 0,scale = 0.8,N = N_sys,
                                                sigma_r_obs = db_10k_FPs['sigma_r_obs'].sample(N_sys,replace=False).to_numpy()).generate_r_obs()
print('Scatter in observations',np.std(r_obs))
print('Scatter in true population',np.std(r_true))
print('None-subzero',(r_true>0).all(),(r_obs>0).all())
# fig,ax = pl.subplots(1,2,figsize=(10,5))
# hist_dict = {'bins':np.linspace(-1,3,41),'density':True,'alpha':0.5}
# ax[0].hist(r_obs,**hist_dict)
# ax[0].hist(r_true,**hist_dict)
# pl.show()
# exit()

assert (r_obs>0).all()
r_obs = jnp.array(r_obs);sigma_r_obs = jnp.array(sigma_r_obs);r_true = jnp.array(r_true)

def j_likelihood_SL(r_obs,sigma_r_obs,key=None,early_return=False):
        # subsample_size = len(r_obs)-1
        sigma_r_obs_2 = sigma_r_obs.flatten()
        r_theory_2_low_lim = 0
        r_theory_2_up_lim = jnp.max(r_obs+5*sigma_r_obs)*jnp.ones(len(r_obs))
        r_theory_2 = numpyro.sample('r_theory_2',dist.Uniform(low = r_theory_2_low_lim, high = r_theory_2_up_lim),sample_shape=(1,)).flatten() 
        alpha_mu = numpyro.deterministic('alpha_mu',0.0)#jnp.squeeze(numpyro.sample("alpha_mu", dist.Uniform(-1,0),sample_shape=(1,),rng_key=key))
        alpha_scale = jnp.squeeze(numpyro.sample("alpha_scale", dist.Uniform(0.2,5),sample_shape=(1,),rng_key=key))
        alpha_s = jnp.squeeze(numpyro.sample("alpha_s", dist.Uniform(0.1,5),sample_shape=(1,)))
        jax.debug.print('Outout: {alpha_s}',alpha_s=alpha_s)
        # jax.debug.print('Shapes {a} {b} {c} {e} {f}',a=r_obs.shape,b=sigma_r_obs_2.shape,c=r_theory_2.shape,
                                                        #  e=alpha_scale.shape,f=alpha_s.shape)
        if early_return: return
        # with numpyro.plate("N", r_theory_2.shape[0], subsample_size=subsample_size):
        #     batch_r_obs = numpyro.subsample(r_obs,event_dim=0)
        #     batch_r_theory_2 = numpyro.subsample(r_theory_2,event_dim=0)
        #     batch_sigma_r_obs_2 = numpyro.subsample(sigma_r_obs_2,event_dim=0)
        #     batch_prob_FP1 = jax_truncnorm.logpdf(x=batch_r_obs,loc=batch_r_theory_2,scale=batch_sigma_r_obs_2,
        #                                             a=-batch_r_theory_2/batch_sigma_r_obs_2,b=np.inf)
        #     batch_prob_FP2 = jax_lognormal(x=batch_r_theory_2,loc=alpha_mu,scale=alpha_scale,s=alpha_s).log_prob()
        #     batch_prob_1 = batch_prob_FP1+batch_prob_FP2
        #     batch_prob_1 = jnp.where(batch_r_theory_2<0,-np.inf,batch_prob_1)
        #     L = numpyro.factor("Likelihood",batch_prob_1)
        # jax.debug.print('Outout: {r_theory_2}',r_theory_2=np.shape(r_theory_2))
        prob_FP1 = jax_truncnorm.logpdf(x=r_obs,loc=r_theory_2,scale=sigma_r_obs_2,
                                                a=-r_theory_2/sigma_r_obs_2,b=np.inf)
        prob_FP2 = jax_lognormal(x=r_theory_2,loc=alpha_mu,scale=alpha_scale,s=alpha_s).log_prob()
        prob_1 = prob_FP1+prob_FP2
        prob_1 = jnp.where(r_theory_2<0,-np.inf,prob_1)
        L = numpyro.factor('Likelihood',prob_1)

# P(r_obs|r)P(r|alpha)P(alpha)

def run_MCMC(batch_bool=False,key_int=seed+2,num_warmup=1000,num_samples=10,num_chains=2,target_accept_prob=0.8,
             warmup_file = './chains/LogNorm_Likelihood_check.csv'):
    N_warmup = len(glob.glob(f"{warmup_file.replace('.csv','')}*"))
    warmup_file = warmup_file.replace('.csv',f'_{N_sys}_{seed}_No_subsamp_Real_Errors_sorted.csv')
    print(f'Saving Warmup to: {warmup_file}')
    model_args = {'r_obs':r_obs,'sigma_r_obs':sigma_r_obs}
    key = jax.random.PRNGKey(key_int)
    outer_kernel = NUTS(model = j_likelihood_SL,target_accept_prob = target_accept_prob,
                        # init_strategy = init_to_value(values={'alpha_s':0.3,'alpha_mu':0.0,'alpha_scale':0.8,'r_theory_2':r_true}),
                        dense_mass=True
                        )
    # outer_kernel = HMCECS(inner_kernel, num_blocks=np.max([N_sys//10,1000]))
    sampler_0 = MCMC(outer_kernel,
                    num_warmup=num_warmup,
                    num_samples=num_samples,
                    num_chains=num_chains,
                    progress_bar=True)
    print('Starting Warmup:')
    sampler_0.warmup(key,**model_args,collect_warmup=True)
    warmup_dict = JAX_samples_to_dict(sampler_0,separate_keys=True)
    db_JAX_warmup = pd.DataFrame(warmup_dict)
    db_JAX_warmup.to_csv(warmup_file,index=False)
    print('Output:',db_JAX_warmup[['alpha_s_0','alpha_s_1','alpha_mu_0','alpha_mu_1','alpha_scale_0','alpha_scale_1']].describe())
    print(f'Saved warmup to {warmup_file}')

run_MCMC()

# import pandas as pd
# import matplotlib.pyplot as pl
# def load_result(file_N):
#     db_JAX = pd.read_csv(f'./chains/LogNorm_Likelihood_check_{file_N}.csv')
#     return db_JAX

# pl.plot(load_result(3)[['alpha_s_0','alpha_s_1','alpha_s_2','alpha_s_3','alpha_s_4']])
# pl.ylim(0,0.4)
# pl.show()

'''
With variable sigma_r_obs:
python3.11-39731.out -> python3.11-39735.out
#With uncertainty increasing with increasing r_true:
python3.11-40466.out -> python3.11-40470.out

'''