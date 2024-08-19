import numpy as np
import pandas as pd
import corner
import matplotlib.pyplot as pl
import sys
from sklearn.neighbors import KernelDensity
from KDEpy import TreeKDE
sys.path.append('/mnt/users/hollowayp/paltas')
from load_h5_file import load_h5_file
from tqdm import tqdm
import pickle
from squash_walkers import squash_walkers
from Save_Summary_Batches import summary_batch

argv = sys.argv
_,cosmo_param = argv

with open('/mnt/extraspace/hollowayp/zBEAMS_data/class_instances/python3.11-Subbatching_0_0-64171.out_10_10_pickle.pkl', 'rb') as f:
    summary_batch_subbatching = pickle.load(f)

if cosmo_param in ['OM','Ode']: X_plot = np.linspace(0,1,10000)
if cosmo_param in ['w_','wa']: X_plot = np.linspace(-3,1,10000)
if cosmo_param in ['alpha_mu_0','alpha_mu_1','alpha_mu_2']: X_plot = np.linspace(0,2,10000)
if cosmo_param in ['alpha_scale_0','alpha_scale_1','alpha_scale_2']: X_plot = np.linspace(0,3,10000)
if cosmo_param in ['alpha_weights_0','alpha_weights_1','alpha_weights_2']: X_plot = np.linspace(0,1,10000)
if cosmo_param in ['s_m']: X_plot = np.linspace(-1,0,10000)
if cosmo_param in ['s_c']: X_plot = np.linspace(0,2,10000)
if cosmo_param in ['scale_m']: X_plot = np.linspace(0,6,10000)
if cosmo_param in ['scale_c']: X_plot = np.linspace(0,2,10000)
OM_samples = squash_walkers(summary_batch_subbatching.JAX_chains_list[0].filter(like=cosmo_param)).to_numpy()
OM_samples_2 = squash_walkers(summary_batch_subbatching.JAX_chains_list[1].filter(like=cosmo_param)).to_numpy()
fig = pl.figure(figsize=(7,5))
for kernel in ['tophat','gaussian']:
    for bandwidth in tqdm([0.01,0.05,0.1,0.5,1.0]):
        kde_OM = KernelDensity(bandwidth=bandwidth,kernel=kernel).fit(np.array(OM_samples))
        OM_score_2 = np.exp(kde_OM.score_samples(OM_samples_2))
        pl.plot(X_plot,np.exp(kde_OM.score_samples(X_plot.reshape(-1,1))),
                label=f'{kernel}, {bandwidth},'+\
                f' Score: {np.round(np.mean(OM_score_2),3)},{np.round(np.median(OM_score_2),3)}')
pl.legend()
pl.title(f'{cosmo_param}',fontsize=18)
pl.savefig(f'/mnt/extraspace/hollowayp/zBEAMS_data/KDE_1D_plots/KDE_1D_{cosmo_param}.png',dpi=500)
pl.close()

'''
for V in alpha_mu_0 alpha_mu_1 alpha_mu_2 alpha_weights_0 alpha_weights_1 alpha_scale_0 alpha_scale_1 alpha_scale_2 s_c s_m scale_c scale_m;
do addqueue -m 10 -q blackhole /mnt/users/hollowayp/python114_archive/bin/python3.11 One_Dimensional_KDE.py $V;
done
'''