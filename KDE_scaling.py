import numpy as np
import matplotlib.pyplot as pl
import pandas as pd
import copy
from sklearn.decomposition import PCA
from sklearn.neighbors import KernelDensity

def calculate_n_PCA(data):
    var_perc = 0.95
    pca_test = PCA(n_components=len(data.columns))  # Reduce to 10 dimensions (adjust based on data)
    pca_test_fit = pca_test.fit_transform(data)
    pca_variance = pca_test.explained_variance_ratio_/np.sum(pca_test.explained_variance_ratio_)
    n_PCA = np.arange(len(data.columns))[np.where(np.cumsum(pca_variance)>var_perc)[0]][0]
    print(f'Using {n_PCA} PCA components, which explain {var_perc}% of the variance')
    return n_PCA

class KDE_scaling:
    def __init__(self,JAX_chains_list_hyp,N_dim,N_sys,N_iterations,bandwidth=0.05,shuffle_params=True,PCA_bool=False):
        np.random.seed(1)
        self.JAX_chains_list_hyp = copy.deepcopy(JAX_chains_list_hyp)
        self.bandwidth = bandwidth
        self.PCA_bool = PCA_bool
        del JAX_chains_list_hyp
        #Rescaling to std = 1:
        self.OM_std = np.mean([np.std(self.JAX_chains_list_hyp[elem]['OM']) for elem in range(len(self.JAX_chains_list_hyp))])
        self.std_array = np.mean([self.JAX_chains_list_hyp[elem].describe().loc['std'].to_numpy() for elem in range(len(self.JAX_chains_list_hyp))],axis=0)
        # print('Std',self.std_array)
        for b_i in range(len(self.JAX_chains_list_hyp)):
            self.JAX_chains_list_hyp[b_i]/=self.std_array
        # print('Describe',self.JAX_chains_list_hyp[0].describe())
        # print('Describe, unscaled',JAX_chains_list_hyp[0].describe())
        self.N_dim = N_dim
        #NOT including OM as this is included seperately later:
        self.param_list = ['Ode','w','wa']+\
                ['alpha_mu_0','alpha_mu_1','alpha_mu_2','alpha_weights_0','alpha_weights_1','alpha_scale_0','alpha_scale_1','alpha_scale_2']+\
                ['s_c','s_m','scale_c','scale_m']
        self.param_list_dict = {}
        self.N_sys = N_sys
        self.N_iterations = N_iterations
        self.kde_dict = {elem:[] for elem in range(len(self.JAX_chains_list_hyp))}
        self.indx_dict = {elem:[] for elem in range(len(self.JAX_chains_list_hyp))}
        self.JAX_chains_list_iterations = {elem:[] for elem in range(len(self.JAX_chains_list_hyp))}
        self.PCA_func = []
        self.PCA_transform = {elem:[] for elem in range(len(self.JAX_chains_list_hyp))}
        self.n_PCA_dict = {}
        for iteration_i in range(N_iterations):
            if shuffle_params: self.param_list_dict[iteration_i] = ['OM']+np.random.choice(self.param_list,N_dim-1,replace=False).tolist()
            else: self.param_list_dict[iteration_i] = ['OM']+self.param_list[:N_dim-1]
            if PCA_bool: self.n_PCA_dict[iteration_i] = calculate_n_PCA(self.JAX_chains_list_hyp[0][self.param_list_dict[iteration_i]])
            for chain in range(len(self.JAX_chains_list_hyp)):
                random_indx = np.random.choice(np.arange(len(self.JAX_chains_list_hyp[chain])),N_sys,replace=False)
                self.indx_dict[chain].append(random_indx)
                self.JAX_chains_list_iterations[chain].append(self.JAX_chains_list_hyp[chain].loc[random_indx])
                if PCA_bool:
                    if chain==0: self.PCA_func.append(PCA(n_components=self.n_PCA_dict[iteration_i],whiten=True))
                    if chain==0: self.PCA_transform[chain].append(self.PCA_func[iteration_i].fit_transform(self.JAX_chains_list_iterations[chain][iteration_i][self.param_list_dict[iteration_i]].to_numpy()))
                    else: self.PCA_transform[chain].append(self.PCA_func[iteration_i].transform(self.JAX_chains_list_iterations[chain][iteration_i][self.param_list_dict[iteration_i]].to_numpy()))
        for iteration_i in range(N_iterations):
            for chain in range(len(self.JAX_chains_list_hyp)):
                if PCA_bool: 
                    kde_input_data = self.PCA_transform[chain][iteration_i]
                    if self.n_PCA_dict[iteration_i]==1: kde_input_data = kde_input_data.reshape(-1,1)
                    # print('KDE input',kde_input_data.shape,np.std(kde_input_data,axis=0))
                else: 
                    kde_input_data = self.JAX_chains_list_iterations[chain][iteration_i][self.param_list_dict[iteration_i]].to_numpy()
                    if self.N_dim==1: kde_input_data = kde_input_data.reshape(-1,1)
                self.kde_dict[chain].append(KernelDensity(bandwidth=bandwidth).fit(kde_input_data))
        # print('Param list:',self.param_list_dict)
    def find_product(self):
        assert len(self.JAX_chains_list_hyp)==2 #Am only finding the product with the 2nd index.
        self.prod_dict = {elem:[] for elem in range(len(self.JAX_chains_list_hyp))}
        for iteration_i in range(self.N_iterations):
            for chain in range(len(self.JAX_chains_list_hyp)):
                if self.PCA_bool:
                    if self.N_dim==1: prod_i = np.exp(self.kde_dict[chain][iteration_i].score_samples(self.PCA_func[iteration_i].transform(self.JAX_chains_list_hyp[1-chain][self.param_list_dict[iteration_i]].to_numpy()).reshape(-1,1)))
                    else: prod_i = np.exp(self.kde_dict[chain][iteration_i].score_samples(self.PCA_func[iteration_i].transform(self.JAX_chains_list_hyp[1-chain][self.param_list_dict[iteration_i]].to_numpy())))
                    # print('Prod shape',prod_i.shape)
                else:
                    if self.N_dim==1: prod_i = np.exp(self.kde_dict[chain][iteration_i].score_samples(self.JAX_chains_list_hyp[1-chain][self.param_list_dict[iteration_i]].to_numpy().reshape(-1,1)))
                    else: prod_i = np.exp(self.kde_dict[chain][iteration_i].score_samples(self.JAX_chains_list_hyp[1-chain][self.param_list_dict[iteration_i]].to_numpy()))
                self.prod_dict[chain].append(prod_i)
        return self
    def compare_1D_histogram(self,plot=False,saveas=None):
        try: self.prod_dict
        except: self.find_product()
        hist_dict_2 = {'bins':np.linspace(0,1,51),'density':True}
        plot_hist_dict_2 = {'bins':np.linspace(-0.1,1.1,51),'density':True,'fill':False,'linewidth':2}
        ratio_list = []
        for iteration_i in range(self.N_iterations):
            for chain_i in range(1):#range(len(self.JAX_chains_list_hyp)): #Degenerate with 1-chain.
                #Applying the std back to the OM:
                w1 = self.prod_dict[1-chain_i][iteration_i]
                w2 = self.prod_dict[chain_i][iteration_i]
                # print('weights',self.JAX_chains_list_hyp[chain_i]['OM'].shape,w1.shape)
                f1,b1 = np.histogram(self.JAX_chains_list_hyp[chain_i]['OM']*self.OM_std,weights=w1,**hist_dict_2)
                f2,b2 = np.histogram(self.JAX_chains_list_hyp[1-chain_i]['OM']*self.OM_std,weights=w2,**hist_dict_2)
                ratio_list.append(np.nanmedian(abs(f1-f2)/f2))
                if plot:
                    print('Iteration',iteration_i,'Params',self.param_list_dict[iteration_i])
                    pl.close()
                    if self.PCA_bool: fig,ax = pl.subplots(figsize=(7,5));ax_0=ax;print('Making figure')
                    else:fig,ax = pl.subplots(1,2,figsize=(10,5));ax_0=ax[0];ax_1=ax[1]
                    ax_0.hist(self.JAX_chains_list_hyp[chain_i]['OM']*self.OM_std,weights=self.prod_dict[1-chain_i][iteration_i],
                            **plot_hist_dict_2,edgecolor='darkorange')
                    ax_0.hist(self.JAX_chains_list_hyp[1-chain_i]['OM']*self.OM_std,weights=self.prod_dict[chain_i][iteration_i],
                            **plot_hist_dict_2,edgecolor='darkblue')
                    ax_0.set_xlabel('OM')
                    ax_0.set_title(f'Product of KDEs in {self.N_dim}D: Bandwidth: {self.bandwidth}')
                    KDE_1D = KernelDensity(bandwidth=self.bandwidth).fit(self.JAX_chains_list_hyp[chain_i]['OM'].loc[self.indx_dict[chain_i][iteration_i]].to_numpy().reshape(-1,1))
                    if not self.PCA_bool:
                        ax_1.hist(self.JAX_chains_list_hyp[chain_i]['OM']*self.OM_std,
                            **plot_hist_dict_2,edgecolor='grey',alpha=0.5)
                        ax_1.hist(self.JAX_chains_list_hyp[1-chain_i]['OM']*self.OM_std,
                            **plot_hist_dict_2,edgecolor='grey',alpha=0.5)
                        ax_1.set_xlabel('OM')
                        ax_1.set_title(f'KDE in 1D: Bandwidth: {self.bandwidth}')
                        X_plot = np.linspace(-0.1,1.1,100)
                        ax_1.plot(X_plot,np.exp(KDE_1D.score_samples(X_plot.reshape(-1,1)/self.OM_std))/self.OM_std,c='k')                    
                    if plot: 
                        if saveas is None: pl.show();print('not saving figure')
                        else:
                            for fmt in ['pdf','png']:
                                fig.savefig(f'{saveas}_Iter_{iteration_i}.{fmt}');print('saving figure');pl.close()
        self.ratio_list = ratio_list
        return self
