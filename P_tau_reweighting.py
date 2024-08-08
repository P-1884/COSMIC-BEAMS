from squash_walkers import squash_walkers
from scipy.stats import gaussian_kde
from emcee import EnsembleSampler
import pandas as pd
import numpy as np
import glob
import sys

argv = sys.argv
python_out_file = argv[1]
batch_i = int(argv[2])
sys_i = int(argv[3])
try: burnin_combo = int(argv[4])
except: burnin_combo = 10000
def sample_single_P_tau_posterior_0(self=None,batch_i=None,sys_i=None,burnin_combo=None,
                                    python_out_file=None,version='latest',N_batch=None):
    print(f"Reweighting the posterior for the {sys_i}th system in the {batch_i}th batch")
    print('Using a burnin for the combined samples of:',burnin_combo)
    print('Using the python out file:',python_out_file)
    print('Using the version:',version)
    if self is None:
        print('Loading files')
        class dummy_class():
            def __init__(self,python_out_file,version):
                if isinstance(version,str):
                    if version=='latest': version = np.max([int(elem.split('_V')[-1].replace('.npy','')) for elem in glob.glob(f'./P_tau_posterior_files/{python_out_file}_pop_hyperparams*')])
                else:
                    assert isinstance(version,int)
                print('Using version number:',version)                
                self.population_hyperparameters = np.load(f'./P_tau_posterior_files/{python_out_file}_pop_hyperparams_V{version}.npy').tolist()
                N_batch = len(glob.glob(f'./P_tau_posterior_files/{python_out_file}_JAX_chains_V{version}_*.csv'))
                print(f'Retrieved {N_batch} batches')
                self.JAX_chains_list = [pd.read_csv(f'./P_tau_posterior_files/{python_out_file}_JAX_chains_V{version}_{k_i}.csv') for k_i in range(N_batch)]
                N_files_combined_samples = len(glob.glob(f'./P_tau_posterior_files/{python_out_file}_combined_inference_samples_V{version}_*.csv'))
                print(f'Retrieved {N_files_combined_samples} walker chains for the combined samples')
                self.combined_inference_samples = {k_ii:pd.read_csv(f'./P_tau_posterior_files/{python_out_file}_combined_inference_samples_V{version}_{k_ii}.csv') for k_ii in range(N_files_combined_samples)}    
                self.N_batch = len(self.JAX_chains_list)
                self.python_out_file = python_out_file
                self.version=version
        self = dummy_class(python_out_file,version)
    # self.P_tau_weights_dict = {b_ii:{sys_ii:None for sys_ii in range(2000)} for b_ii in range(self.N_batch)}
    #Inference of P_tau for the one batch of interest:
    print('KDE 1')
    self.kde_of_batch_of_interest = gaussian_kde(squash_walkers(self.JAX_chains_list[batch_i])[self.population_hyperparameters].T)
    print('KDE 2')
    self.kde_of_batch_with_P_tau = gaussian_kde(squash_walkers(self.JAX_chains_list[batch_i])[self.population_hyperparameters+[f'P_tau_{sys_i}']].T)
    print('KDE 3')
    self.combined_kde_db = pd.concat([self.combined_inference_samples[k_ii].loc[burnin_combo:] for k_ii in self.combined_inference_samples.keys()],ignore_index=True)[self.population_hyperparameters].T
    self.combined_kde = gaussian_kde(self.combined_kde_db)
    print('Finished KDE')
    def reweighted_posterior(combined_kde_db,combined_kde,kde_single,kde_single_with_P_tau):
        ndim = len(combined_kde_db.T.columns)+1
        n_walkers = 2*len(combined_kde_db.T.columns)+2
        def log_prob_fn(x):
            if x[-1]<0: return -np.inf
            if x[-1]>1: return -np.inf
            return kde_single_with_P_tau.logpdf(x)+combined_kde.logpdf(x[:-1])-kde_single.logpdf(x[:-1])
        sampler = EnsembleSampler(nwalkers=n_walkers,ndim=ndim,log_prob_fn=log_prob_fn)
        cur_state_0 = [] 
        print('Generating Prior')
        for col_i in combined_kde_db.T.columns:
            min_hyper_val = np.min(combined_kde_db.T[col_i])
            max_hyper_val = np.max(combined_kde_db.T[col_i])
            cur_state_0.append(np.random.uniform(low=min_hyper_val,high=max_hyper_val,size=(n_walkers,1)))
        cur_state_0.append(np.random.uniform(low=0,high=1,size=(n_walkers,1))) #P_tau
        cur_state = np.concatenate(cur_state_0,axis=1)
        N_steps = 2000
        print('Running Sampler')
        _ = sampler.run_mcmc(cur_state,N_steps,progress=True,skip_initial_state_check=True)
        return sampler
    print('Starting MCMC')
    self.kde_sampler = reweighted_posterior(self.combined_kde_db,self.combined_kde,self.kde_of_batch_of_interest,self.kde_of_batch_with_P_tau)
    # likelihood_db = squash_walkers(self.JAX_chains_list[batch_i])[self.population_hyperparameters]
    # fig = pl.figure(figsize=(10,10))
    # corner.corner(self.combined_kde_db.T[['OM','Ode','alpha_mu_0']],fig=fig,color='k',hist_kwargs={'density':True})
    # corner.corner(likelihood_db[['OM','Ode','alpha_mu_0']],fig=fig,color='blue',hist_kwargs={'density':True})
    # pl.show()
    # self.log_P_tau_weights_dict = {}
    # combined_likelihood = self.combined_kde.logpdf(likelihood_db.T)
    # single_likelihood = self.kde_of_batch_of_interest.logpdf(likelihood_db.T)
    # print('Max comb',np.max(combined_likelihood),combined_likelihood.shape)
    # print('Max sing',np.max(single_likelihood),single_likelihood.shape)
    # self.log_P_tau_weights_dict[batch_i] = combined_likelihood-single_likelihood
    # #Subtract off constant factor (i.e dividing by a constant, as log space):
    # self.log_P_tau_weights_dict[batch_i] -= np.max(self.log_P_tau_weights_dict[batch_i])
    P_tau_reweighted = pd.DataFrame({f'walker_{w_i}':self.kde_sampler.get_chain()[:,w_i,-1] for w_i in range(self.kde_sampler.get_chain().shape[1])})
    P_tau_reweighted.to_csv(f'./P_tau_posterior_files/{python_out_file}_V{self.version}_B{batch_i}_S{sys_i}.csv',index=False)
    return self

sample_single_P_tau_posterior_0(self=None,batch_i=batch_i,sys_i=sys_i,
                                burnin_combo = burnin_combo,
                                python_out_file=python_out_file)
