import numpy as np
import matplotlib.pyplot as pl
import pandas as pd
# import dynesty
# from pathos.pools import ProcessPool as Pool
# from multiprocessing import cpu_count
# from KDEpy import TreeKDE
from sklearn.neighbors import KernelDensity
from emcee import PTSampler
from tqdm import tqdm
import h5py
import multiprocessing
from run_ptemcee import run_ptemcee, gaussian_kde_product_log_prob, log_prior
import sys
ntemps = int(sys.argv[1])

### THIS CODE MUST BE RUN IN THE python3_emcee environment, which uses an older version of emcee.
print('Loading data')
db = pd.read_csv('./test_data_for_download.csv')
db.drop('Unnamed: 0', axis=1, inplace=True)

db = db[['OM',
         'Ode','w','wa',
    'alpha_scale_0','alpha_scale_1',
    'alpha_scale_2',
    'alpha_mu_0','alpha_mu_1','alpha_mu_2',
    'alpha_weights_0','alpha_weights_1',
    's_c','s_m',
    'scale_c','scale_m']]

n_params = len(db.columns)
range_dict = {elem: [db[elem].min(), db[elem].max()] for elem in db.columns}
range_min_array = np.array([range_dict[elem][0] for elem in db.columns])
range_max_array = np.array([range_dict[elem][1] for elem in db.columns])

db = ((db-range_min_array)/(range_max_array-range_min_array)) #Rescaling to [0,1]

print('Fitting GMM to data')
bandwidth=0.01
kernel='gaussian'
N_comp = 1000

type='KDE'
# kde = TreeKDE(kernel=kernel, bw=bandwidth).fit(data=db.to_numpy())
kde = KernelDensity(kernel=kernel, bandwidth=bandwidth).fit(db.to_numpy())

sampler = run_ptemcee(db,kde,ntemps=ntemps,n_burnin=1000,n_steps=10000,n_thin=1)
# pd.DataFrame(sampler.chain).to_csv('./ptemcee_chain.csv')

hf = h5py.File(f'/mnt/extraspace/hollowayp/zBEAMS_data/ptemcee_samples/ptemcee_chain_{ntemps}_{n_params}.h5', 'w')
hf.create_dataset('data', data=sampler.chain)
hf.close()

print(ntemps,n_params,'Acceptance Fraction',sampler.tswap_acceptance_fraction)

# addqueue -m 10 /mnt/users/hollowayp/python3_emcee/bin/python3.11 ./ptemcee_sampling.py