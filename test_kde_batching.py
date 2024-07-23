from scipy.stats import gaussian_kde
from scipy.integrate import trapezoid
from emcee import EnsembleSampler
from numpyro_truncnorm_GMM_fit import numpyro_truncnorm_GMM_fit
from GMM_class import GMM_class
import numpy as np
import pandas as pd
import matplotlib.pyplot as pl
from tqdm import tqdm
import time
class test_kde_batching():
    def __init__(self,N_data,N_batch,truth_dict,inf_if_weights_unordered=False):
        self.N_data = N_data
        self.N_batch = N_batch
        ground_truth_dict = {'list_of_mu':np.array([truth_dict[0]['mu'],truth_dict[1]['mu'],truth_dict[2]['mu']]),
                             'list_of_sigma':np.array([truth_dict[0]['sigma'],truth_dict[1]['sigma'],truth_dict[2]['sigma']]),
                             'list_of_weights':np.array([truth_dict[0]['weight'],truth_dict[1]['weight'],truth_dict[2]['weight']])}
        self.ground_truth_dict = ground_truth_dict
        self.data = GMM_class(**ground_truth_dict).sample(trunc_at_zero=True,N_samples=N_data)
        self.data = np.random.choice(self.data,size=N_data,replace=False) #Random ordering
        data_batches = np.array_split(self.data,N_batch)
        self.data_batches = {elem:data_batches[elem] for elem in range(N_batch)}
        #Now fitting a truncated normal to these data batches:
        self.MCMC_fits = {elem:numpyro_truncnorm_GMM_fit(self.data_batches[elem],
                                                         N_comp = len(ground_truth_dict['list_of_mu']),
                                                         num_warmup=20000,num_samples=20000,
                                                         return_all_samples=True,
                                                         inf_if_weights_unordered=inf_if_weights_unordered) for elem in range(N_batch)}
        self.MCMC_fits_all_data = numpyro_truncnorm_GMM_fit(self.data,
                                                         N_comp = len(ground_truth_dict['list_of_mu']),
                                                         num_warmup=20000,num_samples=20000,
                                                         return_all_samples=True,
                                                         inf_if_weights_unordered=inf_if_weights_unordered)
        return
    def order_weights(self,MCMC_dict):
        self.MCMC_db_i = {}
        for c_i in range(len(self.ground_truth_dict['list_of_mu'])):
            self.MCMC_db_i[f'alpha_mu_{c_i}'] = MCMC_dict[f'alpha_mu_{c_i}'].flatten()
            self.MCMC_db_i[f'alpha_scale_{c_i}'] = MCMC_dict[f'alpha_scale_{c_i}'].flatten()
            self.MCMC_db_i[f'alpha_weights_{c_i}'] = MCMC_dict['alpha_weights'][:,c_i]
        self.MCMC_db_ii = pd.DataFrame(self.MCMC_db_i)
        #Putting in weight order:
        MCMC_db_weight_order = np.argsort(self.MCMC_db_ii[[elem for elem in self.MCMC_db_ii.columns if 'weight' in elem]].median(axis=0))
        MCMC_db_weight_order = np.array([[f'alpha_mu_{elem}',f'alpha_scale_{elem}',f'alpha_weights_{elem}'] for elem in np.array([0,1,2])[MCMC_db_weight_order]]).flatten().tolist()
        self.MCMC_db_ii = self.MCMC_db_ii[MCMC_db_weight_order].to_numpy()
        ordered_column_names = np.array([[f'alpha_mu_{elem}',f'alpha_scale_{elem}',f'alpha_weights_{elem}'] for elem in np.array([0,1,2])]).flatten().tolist()
        return pd.DataFrame(self.MCMC_db_ii,columns = ordered_column_names)
    def find_kde_product(self,MCMC_db,N_batch,N_steps):
            kde = {elem:gaussian_kde(MCMC_db[elem].T) for elem in range(N_batch)}
            def gaussian_kde_product_log_prob(x):
                if x[2]<0 or x[2]>1: return -np.inf
                if x[5]<0 or x[5]>1: return -np.inf
                if x[8]<0 or x[8]>1: return -np.inf
                if (x[2]>x[5]) or (x[5]>x[8]): return -np.inf #Asserting weight ordering - seems a bit dodge?
                log_prob =  np.sum(np.array([kde[elem].logpdf(x) for elem in range(N_batch)]),axis=0)
                return log_prob
            n_walkers = 20
            sampler = EnsembleSampler(nwalkers=n_walkers,ndim=9,log_prob_fn=gaussian_kde_product_log_prob)
            cur_state_0 = [] 
            cur_simplex_0 = np.random.dirichlet([1]*3,size=(n_walkers,1))
            #Putting weights in order:
            cur_simplex = np.nan*np.zeros(cur_simplex_0.shape)
            cur_simplex[:,0,0] = np.min(cur_simplex_0,axis=2).flatten()
            cur_simplex[:,0,1] = np.median(cur_simplex_0,axis=2).flatten()
            cur_simplex[:,0,2] = np.max(cur_simplex_0,axis=2).flatten()
            w_i=0
            for col_i in MCMC_db[0].columns:
                w_ii = cur_simplex.T[w_i].T
                if 'weight' in col_i: cur_state_0.append(w_ii);w_i+=1
                else: cur_state_0.append(np.random.uniform(low=0.01,high=4,size=(n_walkers,1)))
            cur_state = np.concatenate(cur_state_0,axis=1)
            _ = sampler.run_mcmc(cur_state,N_steps,progress=True,skip_initial_state_check=True)
            return sampler,kde
    def return_stuff(self,N_steps = 100):
        self.MCMC_db = {}
        for batch_ii in range(self.N_batch):
            self.MCMC_db[batch_ii] = self.order_weights(self.MCMC_fits[batch_ii])
        self.MCMC_db_all = self.order_weights(self.MCMC_fits_all_data)
        print('ALL',type(self.MCMC_db_all))
        print("Batch",type(self.MCMC_db),type(self.MCMC_db[0]))
        self.sampler,self.kde = self.find_kde_product(self.MCMC_db,N_batch=self.N_batch,N_steps=N_steps)
        # self.meshgrid_list = []
        # self.meshgrid_db_list = []
        # for batch_i in tqdm(range(self.N_batch)):
        #     for point_i in range(N_random_points):
        #         meshgrid_list_i = []
        #         for column_i in self.MCMC_db[batch_i].columns:
        #             if 'mu' in column_i: meshgrid_list_i.append(np.random.uniform(0,5))
        #             if 'sigma' in column_i: meshgrid_list_i.append(np.random.uniform(0.01,2))
        #             if 'weights' in column_i: meshgrid_list_i.append(np.random.uniform(0,1))
        #         self.meshgrid_db_list.append(meshgrid_list_i)
        # self.meshgrid_db_list = np.array(self.meshgrid_db_list).T
        # self.kde_eval = np.sum(np.array([self.kde[elem].logpdf(self.meshgrid_db_list) for elem in range(self.N_batch)]),axis=0)
        # self.meshgrid_db_list = pd.DataFrame(self.meshgrid_db_list.T,
        #                                      columns = self.MCMC_db[0].columns)
        # self.meshgrid_db_list['PDF'] = self.kde_eval
        return self
        # assert X_eval.shape==self.kde_eval.shape
        # kde_normalisation = trapezoid(y=self.kde_eval,x=X_eval)
        # self.kde_eval/=kde_normalisation
        # pl.plot(X_eval,self.kde_eval)
        # for elem in range(self.N_batch):
        #     pl.plot(X_eval,self.kde[elem].pdf(X_eval),alpha=0.5,c='k')
        # pl.hist(self.data,density=True,bins=50)
        # pl.show()

truth_dict = {0:{'mu':0.5,'sigma':0.1,'weight':0.1},
            1:{'mu':2,'sigma':0.5,'weight':0.2},
            2:{'mu':3.1,'sigma':0.3,'weight':0.7}}

kde_class_0 = test_kde_batching(60000,4,truth_dict,inf_if_weights_unordered=True)
kde_class_1 = kde_class_0.return_stuff(N_steps = 40000)

def order_weights_0(MCMC_dict,N_comp):
        MCMC_db_i = {}
        for c_i in range(N_comp):
            MCMC_db_i[f'alpha_mu_{c_i}'] = MCMC_dict[f'alpha_mu_{c_i}'].to_numpy().flatten()
            MCMC_db_i[f'alpha_scale_{c_i}'] = MCMC_dict[f'alpha_scale_{c_i}'].to_numpy().flatten()
            MCMC_db_i[f'alpha_weights_{c_i}'] = MCMC_dict[f'alpha_weights_{c_i}'].to_numpy().flatten()
        MCMC_db_ii = pd.DataFrame(MCMC_db_i)
        #Putting in weight order:
        MCMC_db_weight_order = np.argsort(MCMC_db_ii[[elem for elem in MCMC_db_ii.columns if 'weight' in elem]].median(axis=0))
        MCMC_db_weight_order = np.array([[f'alpha_mu_{elem}',f'alpha_scale_{elem}',f'alpha_weights_{elem}'] for elem in np.array([0,1,2])[MCMC_db_weight_order]]).flatten().tolist()
        MCMC_db_ii = MCMC_db_ii[MCMC_db_weight_order].to_numpy()
        ordered_column_names = np.array([[f'alpha_mu_{elem}',f'alpha_scale_{elem}',f'alpha_weights_{elem}'] for elem in np.array([0,1,2])]).flatten().tolist()
        return pd.DataFrame(MCMC_db_ii,columns = ordered_column_names)


MCMC_batch_data_dict = {}
for chain_i in range(kde_class_1.sampler.get_chain().shape[1]):
    db_ii = pd.DataFrame(kde_class_1.sampler.get_chain()[:,chain_i,:],
                         columns=kde_class_1.MCMC_db_all.columns)
    MCMC_batch_data_dict[chain_i] = order_weights_0(db_ii,3)

file_out = f'./kde_batch_testing/batch_test_{np.round(time.time(),2)}'
print(f'Saving file to: {file_out}')
for chain_i in range(len(MCMC_batch_data_dict.keys())):
    MCMC_batch_data_dict[chain_i].to_csv(f'{file_out}_chain_{chain_i}.csv',index=False)

for batch_i in range(len(kde_class_1.MCMC_db.keys())):
    kde_class_1.MCMC_db[batch_i].to_csv(f'{file_out}_batch_{batch_i}.csv',index=False)

kde_class_1.MCMC_db_all.to_csv(f'{file_out}_all.csv',index=False)

