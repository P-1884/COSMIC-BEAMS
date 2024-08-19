import numpy as np
import matplotlib.pyplot as pl
import pandas as pd
from sklearn.neighbors import KernelDensity
from emcee import PTSampler
from tqdm import tqdm
import h5py
import multiprocessing
### THIS CODE MUST BE RUN IN THE python3_emcee environment, which uses an older version of emcee.

def gaussian_kde_product_log_prob(x,kde):
    if np.any(x<0) or np.any(x>1):
        return -np.inf
    log_prob = kde.score([x])
    return log_prob

def log_prior(x):
    return 0

def run_ptemcee(db,kde,ntemps=20,n_burnin=1000,n_steps=10000,n_thin=10):
    '''
    Uses uniform prior, and assumes db values are already scaled to [0,1]
    '''
    ntemps = ntemps
    nwalkers = 2*len(db.columns)+2
    ndim = len(db.columns)
    with multiprocessing.Pool() as pool:
        sampler=PTSampler(ntemps, nwalkers, ndim, gaussian_kde_product_log_prob, log_prior,pool=pool,loglargs = (kde,))
        p0 = np.random.uniform(low=0, high=1.0, size=(ntemps, nwalkers, ndim))
        for p, lnprob, lnlike in tqdm(sampler.sample(p0, iterations=n_burnin)):
            pass
        sampler.reset()
        for p, lnprob, lnlike in tqdm(sampler.sample(p, lnprob0=lnprob,
                                                lnlike0=lnlike,
                                                iterations=n_steps, thin=n_thin)):
            pass
    return sampler
