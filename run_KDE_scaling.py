from sklearn.neighbors import KernelDensity
import sys
sys.path.append('/mnt/users/hollowayp/paltas')
from KDE_one_stop import drop_extra_columns,hyperparam_range_dict,one_stop_kde
from Save_Summary_Batches import summary_batch
from squash_walkers import squash_walkers
from load_h5_file import load_h5_file
import matplotlib.pyplot as pl
from KDEpy import TreeKDE
import pandas as pd
import numpy as np
import pickle
import corner
from tqdm.notebook import tqdm
from sklearn.neighbors import KernelDensity
from sklearn.decomposition import PCA
from KDE_scaling import KDE_scaling,calculate_n_PCA

JAX_chains_list_hyp = [pd.read_csv(f'/mnt/extraspace/hollowayp/zBEAMS_data/cmd_outputs/python3.11-Subbatching_cp_total_{b_i}.out') for b_i in range(2)]
bandwidth=0.1
PCA_bool=True
KDE_scaling(JAX_chains_list_hyp,N_dim=16,
            N_sys=len(JAX_chains_list_hyp[0]),N_iterations=5,
            bandwidth=bandwidth,PCA_bool=PCA_bool).compare_1D_histogram(plot=True,saveas=f'./KDE_scaling_16D_400k_samples_bw{bandwidth}_PCA_{PCA_bool}').ratio_list

#addqueue -m 6 -c '30hr' -g 'KDE_Scaling' /mnt/users/hollowayp/python114_archive/bin/python3.11 ./run_KDE_scaling.py
