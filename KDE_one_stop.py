import pickle
from Save_Summary_Batches import summary_batch
from squash_walkers import squash_walkers
from tqdm import tqdm
from plot_JAX_corner import label_dict
import numpy as np
import pandas as pd
import corner
import matplotlib.pyplot as pl
import sys
from sklearn.neighbors import KernelDensity
from multiprocessing import Pool


def drop_extra_columns(db,P_tau=True,zL=True,zS=True,Ok=False,alpha_weights_2=False):
    columns_to_drop = []
    if P_tau: columns_to_drop += list(db.filter(regex='P_tau'))
    if zL: columns_to_drop += list(db.filter(regex='zL'))
    if zS: columns_to_drop += list(db.filter(regex='zS'))
    if Ok: columns_to_drop += list(db.filter(regex='Ok'))
    if alpha_weights_2: columns_to_drop += list(db.filter(regex='alpha_weights_2'))
    return db[db.columns.drop(columns_to_drop)]

hyperparam_range_dict = {
    'OM':(0,1),
    'Ode':(0,1),
    'w':(-3,1),
    'wa':(-3,1),
    's_m':(-1,0),
    's_c':(0,1),
    'scale_m':(0,1),
    'scale_c':(0,2),
    'alpha_mu_0':(0,2),
    'alpha_mu_1':(0,2),
    'alpha_mu_2':(0,2),
    'alpha_scale_0':(0,2),
    'alpha_scale_1':(0,2),
    'alpha_scale_2':(0,2),
    'alpha_weights_0':(0,1),
    'alpha_weights_1':(0,1),
    'alpha_weights_2':(0,1)
} 

scaling_dict = {
    'OM':(0.01,0.01),
    'w':(0.05,0.1),
    'Ode':(0.01,0.05),
    'wa':(0.05,0.05),
    's_c':(0.01,0.05),
    's_m':(0.05,0.05),
    'scale_m':(0.05,0.1),
    'scale_c':(0.05,0.05),
    'alpha_mu_0':(0.01,0.01),
    'alpha_mu_1':(0.01,0.01),
    'alpha_mu_2':(0.01,0.01),
    'alpha_weights_0':(0.01,0.01),
    'alpha_weights_1':(0.01,0.01),
    'alpha_scale_0':(0.01,0.01),
    'alpha_scale_1':(0.01,0.01),
    'alpha_scale_2':(0.01,0.01)}


class one_stop_kde():
    def __init__(self,JAX_chain_dict,bandwidth = 0.5,kernel='gaussian',metric='infinity',optimal_scaling = False,scaling_bool=np.nan):
        #JAX_chains_dict walkers should be squashed already.
        self.JAX_chain_dict = JAX_chain_dict
        if optimal_scaling:
            self.scaling = np.array([scaling_dict[elem][scaling_bool] for elem in JAX_chain_dict[0].columns])
        else:
            self.scaling = np.array([np.mean([np.std(JAX_chain_dict[k_i][elem]) for k_i in JAX_chain_dict.keys()]) for elem in JAX_chain_dict[0].columns])
        self.scaled_JAX_chain_dict = {k_i:self.JAX_chain_dict[k_i]/self.scaling for k_i in JAX_chain_dict.keys()}
        self.kernel = kernel
        self.metric = metric
        self.bandwidth = bandwidth
    def determine_kde(self):
        print('Starting KDE Determination')
        self.kde_dict = {k_i:KernelDensity(bandwidth=self.bandwidth,kernel=self.kernel,
                                           metric=self.metric).fit(self.scaled_JAX_chain_dict[k_i].to_numpy())
                                           for k_i in tqdm(self.scaled_JAX_chain_dict.keys())}
        print('Finished KDE Determination')
        return self
    def score_samples(self,input_tuple):
        k_y,kde,chains = input_tuple
        return k_y,np.exp(kde.score_samples(chains[k_y].to_numpy()))
    def calculate_weights(self):
        try: self.kde_dict
        except: self.determine_kde()
        print('Starting Weight Calculation')
        self.weights_dict = {}
        for k_x in tqdm(self.scaled_JAX_chain_dict.keys()):
            self.weights_dict[k_x] = {}
            # k_y_list = list(self.scaled_JAX_chain_dict.keys())
            # k_y_list.remove(k_x)
            # input_tuples = [(k_y,self.kde_dict[k_x],self.scaled_JAX_chain_dict) for k_y in k_y_list]
            # with Pool() as p:
            #     results = list(tqdm(p.imap(self.score_samples,input_tuples),total=len(k_y_list)))
            # for k_y,weights_i in results:
            #     self.weights_dict[k_x][k_y] = weights_i
            #     self.weights_dict[k_x][k_y]/=np.sum(self.weights_dict[k_x][k_y])
            for k_y in self.scaled_JAX_chain_dict.keys():
                if k_x==k_y: continue
                #Weights for the **k_y** samples, which should therefore be applied to the **k_y** dataset.
                self.weights_dict[k_x][k_y] = np.exp(self.kde_dict[k_x].score_samples(self.scaled_JAX_chain_dict[k_y].to_numpy()))
                #Each kde should have the same overall effect, so for a given k_y, each kde_kx should have the same sum of weights.
                #For simplicity, will just make all weight entries sum to 1.
                self.weights_dict[k_x][k_y]/=np.sum(self.weights_dict[k_x][k_y])
        print('Finished Weight Calculation')
        return self


try:
    argv = sys.argv
    print('argv',argv)
    _,bandwidth,kernel,metric,optimal_scaling,scaling_bool = argv
    if optimal_scaling=='True': bandwidth=4#1.0 #Optimal scaling rescales the data, so optimal bandwidth should be sqrt(N_dim) = 4??
    with open('/mnt/extraspace/hollowayp/zBEAMS_data/class_instances/python3.11-Subbatching_0_0-64171.out_10_10_pickle.pkl', "rb") as input_file:
        summary_batch = pickle.load(input_file)
    test_data_dict = {}
    for chain_i in tqdm(range(10)):
        test_data_dict[chain_i] = squash_walkers(drop_extra_columns(summary_batch.JAX_chains_list[chain_i],#.loc[:1999],
                                                                    Ok=True,alpha_weights_2=True))
    osk = one_stop_kde(test_data_dict,bandwidth=float(bandwidth),kernel=kernel,metric=metric,
                       optimal_scaling=eval(optimal_scaling),
                       scaling_bool=eval(scaling_bool)).calculate_weights()
    # osk_0 = plot_kde_corner(osk_0,only_cosmo=True)
    out_file = f'/mnt/extraspace/hollowayp/zBEAMS_data/class_instances/one_stop_kde_subbatch_{metric}_{kernel}_{bandwidth}_{optimal_scaling}_{scaling_bool}.pkl'
    print('Saving to:',out_file)
    with open(out_file, 'wb') as file:
        pickle.dump(osk, file) 
except Exception as ex:
    print('Exception:',ex)

'''
for metric in euclidean infinity;
do for kernel in gaussian tophat exponential; 
do for bandwidth in 0.001 0.01 0.05 0.1 0.2 0.5 1.0; 
do addqueue -m 10 -q blackhole -g $metric$kernel$bandwidth /mnt/users/hollowayp/python114_archive/bin/python3.11 ./KDE_one_stop.py $bandwidth $kernel $metric;
done; done; done

for metric in euclidean infinity;
do for kernel in gaussian; 
do for bandwidth in 1.0; 
do for scaling_bool in 0 1;
do addqueue -m 10 -q blackhole -g $metric$kernel$bandwidth /mnt/users/hollowayp/python114_archive/bin/python3.11 ./KDE_one_stop.py $bandwidth $kernel $metric True $scaling_bool;
done; done; done;done
'''