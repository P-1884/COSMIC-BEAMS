print('Loading Packages')
import os
os.environ['JAX_PLATFORMS'] = 'cpu'
# os.environ['JAX_ENABLE_X64']='True'
import jax
# jax.config.update("jax_enable_x64", True)

from zbeamsfunctions_SL import likelihood_SL,likelihood_spec_contam_SL,likelihood_phot_contam_SL,likelihood_phot_SL,r_SL
from astropy.cosmology import LambdaCDM,FlatLambdaCDM,wCDM,FlatwCDM,w0waCDM
from zbeamsfunctions import mu_w,likelihood,likelihood_spec
from mcmcfunctions_SL_JAX import j_likelihood_SL,run_MCMC
from Lenstronomy_Cosmology import Background, LensCosmo
from scipy.stats import multivariate_normal as MVN
from sklearn.mixture import GaussianMixture as GMM
from mcmcfunctions import mcmc,mcmc_spec,mcmc_phot
from numpyro.infer import MCMC, NUTS, HMC, HMCECS
from numpyro import distributions as dist, infer
from squash_walkers import squash_walkers
from scipy.stats import truncnorm, norm
from numpyro.diagnostics import summary
import matplotlib.patches as mpatches
from mcmcfunctions_SL import mcmc_SL
import matplotlib.lines as mlines
from cosmology_JAX import j_r_SL
from jax import random,grad, jit
import matplotlib.pyplot as pl
from jax.random import PRNGKey
from importlib import reload
from subprocess import run
import jax.numpy as jnp
import jax_cosmo as jc
from tqdm import tqdm
import scipy.sparse
import pandas as pd
#import arviz as az #No longer compatible with scipy version 1.13.0 (previously used scipy version 1.11.4)
import numpy as np
import importlib
import numpyro
import corner
import emcee
import time
import glob
import sys
import jax
import os
from plot_JAX_chains import plot_JAX_chains
from retrieve_chain_files import retrieve_chain_files
from locate_latest_chains import locate_latest_chains
from Summary_Plots_Class import summary_plots
from plot_JAX_corner import plot_JAX_corner
try:importlib.reload(sys.modules['mcmcfunctions_SL'])
except Exception as ex: print(f'Cannot reload: {ex}')
from mcmcfunctions_SL import mcmc_SL
from numpyro_truncnorm_GMM_fit import numpyro_truncnorm_GMM_fit
from plot_JAX_corner import plot_JAX_corner,percentile_str,plot_mu_sig,range_dict,label_dict

Om_fid = 0.3;Ode_fid = 0.7;H0_fid = 70;w_fid = -1.0;wa_fid=0

cosmo_type = 'wCDM'
'''
Have shown the JAX and emcee modules give answers in agreement for a very simple cosmology. Need to further
demonstrate this with more complex cosmologies (inc w0wa cosmology which emcee doesn't yet have?), but most
importantly including contamination + photometry.
'''

from convert_ipynb_to_py import save_notebook_as_python_file 
def save_backup():
    # Saves code each time it is run:
    N_code_backups = np.max([len(glob.glob('./code_backups/mcmcfunctions_SL_JAX*')),
                            len(glob.glob('./code_backups/zBEAMS_Application_to_Strong_Lensing.py*'))])
    code_backup_time = np.round(time.time(),4)
    notebook_backup_file = f'./code_backups/zBEAMS_Application_to_Strong_Lensing_{N_code_backups}_{code_backup_time}.py'
    print(f'Saving notebook backup to {notebook_backup_file}')
    save_notebook_as_python_file('./zBEAMS_Application_to_Strong_Lensing.ipynb',notebook_backup_file)

save_backup()


from sklearn.neighbors import KernelDensity
import numpy as np
def rescale_kde(input_data,kernel='tophat',bandwidth=1.0):
    print(input_data.shape)
    mean_vals = np.mean(input_data,axis=0)
    std_vals = np.std(input_data,axis=0)
    rescaled_data = (input_data-mean_vals)/std_vals
    rescaled_KDE_samples = pd.DataFrame(KernelDensity(kernel=kernel,bandwidth=bandwidth).fit(rescaled_data).sample(100000),columns = som1.population_hyperparameters)
    KDE_samples = rescaled_KDE_samples*std_vals + mean_vals
    return KDE_samples

def rescale_func(input_data):
    # assert False #Need to run rescaling across batches, not per batch.
    mean_vals = np.mean(input_data,axis=0)
    std_vals = np.std(input_data,axis=0)
    rescaled_data = (input_data-mean_vals)/std_vals
    # rescaled_KDE_samples = pd.DataFrame(KernelDensity(kernel='tophat').fit(rescaled_data).sample(100000),columns = som1.population_hyperparameters)
    rescale_dict = {'mu':mean_vals.to_dict(),'std':std_vals.to_dict()}
    return rescale_dict

from squash_walkers import squash_walkers
from importlib import reload
import sys
# reload(sys.modules['test_kde_batching'])
from test_kde_batching import find_kde_product_reweighting
import pickle
import pandas as pd
from Save_Summary_Batches import summary_batch

import sys
argv = sys.argv
kernel = argv[1]
bandwidth = float(argv[2])
try:
    p_value = eval(argv[3])
except:
    p_value = 2.0

print(f'kernel: {kernel}, bandwidth: {bandwidth}', 'p_value:',p_value)

# with open('/mnt/extraspace/hollowayp/zBEAMS_data/class_instances/kde_test_0.1.pkl','rb') as file_00: 
with open('/mnt/extraspace/hollowayp/zBEAMS_data/class_instances/python3.11-Fiducial_0_0-63921.out_5_10_pickle.pkl','rb') as file_00: 
    som1 = pickle.load(file_00)

squashed_walkers = squash_walkers(som1.JAX_chains_list[0]) #Only selecting first batch for speed
bounded_dict = {elem:(squashed_walkers[elem].min(),squashed_walkers[elem].max()) 
                                                for elem in som1.population_hyperparameters}
kde_prod,_ = find_kde_product_reweighting([squash_walkers(som1.JAX_chains_list[0].loc[0:1999])],1,20000,
                                ['alpha'],som1.population_hyperparameters,
                                rescaling_dict = rescale_func(squash_walkers(som1.JAX_chains_list[0])[som1.population_hyperparameters]),
                                # bounded_dict={'OM':(0,1),'Ode':(0,1),'wa':(-3,1)}
                                bounded_dict = bounded_dict,
                                kernel=kernel,bandwidth=bandwidth,
                                p_value=p_value)


for k_i in kde_prod.keys():
    if p_value==2.0:
        file_out = f'/mnt/extraspace/hollowayp/zBEAMS_data/KDE_tests/kde_prod_{kernel}_{bandwidth}_{k_i}.csv'
    else:
        file_out = f'/mnt/extraspace/hollowayp/zBEAMS_data/KDE_tests/kde_prod_{kernel}_{bandwidth}_{p_value}_{k_i}.csv'
    print('Outfile',file_out)
    kde_prod[k_i].to_csv(file_out)

# som1.plot_JAX_corner(kde_prod[0],burnin_combo=2000)