#from tensorflow.python.client import device_lib
import sys
import distutils
# argv = sys.argv
# print('ARGS',argv)
#import os
#os.environ["JAX_ENABLE_X64"] = 'True'
import numpyro
# numpyro.set_platform(platform='gpu')
import jax
import time
from jax import local_device_count,default_backend,devices
# if default_backend()=='cpu': => Doesn't seem to recognise more than one device - always just prints jax.devices=1.
#     numpyro.util.set_host_device_count(4)
#     print('Device count:', len(jax.devices()))

# def get_available_gpus():
#     local_device_protos = device_lib.list_local_devices()
#     return [x.name for x in local_device_protos if x.device_type == 'GPU']
# print('GPU',get_available_gpus())
from zbeamsfunctions_SL import likelihood_SL,likelihood_spec_contam_SL,likelihood_phot_contam_SL,likelihood_phot_SL,r_SL
from astropy.cosmology import LambdaCDM,FlatLambdaCDM,wCDM,FlatwCDM,w0waCDM
from zbeamsfunctions import mu_w,likelihood,likelihood_spec
from mcmcfunctions_SL_JAX import j_likelihood_SL,run_MCMC
from Lenstronomy_Cosmology import Background, LensCosmo
from JAX_samples_to_dict import JAX_samples_to_dict
from mcmcfunctions import mcmc,mcmc_spec,mcmc_phot
from numpyro import distributions as dist, infer
from numpyro.infer import MCMC, NUTS, HMC
import matplotlib.patches as mpatches
from mcmcfunctions_SL import mcmc_SL
from scipy.stats import truncnorm
import matplotlib.lines as mlines
from cosmology_JAX import j_r_SL
from jax import random,grad,jit
import matplotlib.pyplot as pl
import jax.numpy as jnp
import jax_cosmo as jc
from tqdm import tqdm
import scipy.sparse
import pandas as pd
#import arviz as az
import numpy as np
import importlib
import corner
import emcee
import time
import glob
import sys
print('COMPS',local_device_count(),default_backend(),devices())
import argparse