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
from sklearn.mixture import GaussianMixture as GMM
from KDEpy import NaiveKDE,TreeKDE,FFTKDE
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

# truth_dict = {0:{'mu':0.5,'sigma':0.1,'weight':0.1},
# 			1:{'mu':2,'sigma':0.5,'weight':0.2},
# 			2:{'mu':3.1,'sigma':0.3,'weight':0.7}}

# kde_class_0 = test_kde_batching(60000,4,truth_dict,inf_if_weights_unordered=True)
# kde_class_1 = kde_class_0.return_stuff(N_steps = 40000)

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


# MCMC_batch_data_dict = {}
# for chain_i in range(kde_class_1.sampler.get_chain().shape[1]):
# 	db_ii = pd.DataFrame(kde_class_1.sampler.get_chain()[:,chain_i,:],
# 						 columns=kde_class_1.MCMC_db_all.columns)
# 	MCMC_batch_data_dict[chain_i] = order_weights_0(db_ii,3)

# file_out = f'./kde_batch_testing/batch_test_{np.round(time.time(),2)}'
# print(f'Saving file to: {file_out}')
# for chain_i in range(len(MCMC_batch_data_dict.keys())):
# 	MCMC_batch_data_dict[chain_i].to_csv(f'{file_out}_chain_{chain_i}.csv',index=False)

# for batch_i in range(len(kde_class_1.MCMC_db.keys())):
# 	kde_class_1.MCMC_db[batch_i].to_csv(f'{file_out}_batch_{batch_i}.csv',index=False)

# kde_class_1.MCMC_db_all.to_csv(f'{file_out}_all.csv',index=False)


class kde_posterior_batching:
	def __init__(self,list_of_data_samples,list_of_list_of_weight_hyperparameters,bw_factor=None): # Nested lists of separate weight hyperparameters
		self.N_batch = len(list_of_data_samples)
		print('N_batch',self.N_batch)
		self.list_of_data_samples = list_of_data_samples
		self.list_of_reweighted_data_samples = [(elem-rescaling_dict['mu'])/rescaling_dict[std] for elem in range(len(self.list_of_data_samples))]
		# print('list of data samples',self.list_of_data_samples)
		self.hyperparameters = self.list_of_data_samples[0].columns
		# print('Hyperparameters',self.hyperparameters)
		self.N_hyperparameters = len(self.hyperparameters)
		# print('N_hyperparameters',self.N_hyperparameters)
		self.weight_hyperparameters = list_of_list_of_weight_hyperparameters
		# print('Weight hyperparameters:',self.weight_hyperparameters)
		self.weight_hyperparameters_flat = [elem for sublist in self.weight_hyperparameters for elem in sublist]
		self.nonweight_hyperparameters = [elem for elem in self.hyperparameters if elem not in self.weight_hyperparameters_flat]
		# print('Nonweight hyperparameters:',self.nonweight_hyperparameters)
		assert isinstance(list_of_data_samples,list)
		print('Iterating through distinct weight hyperparameters:')
		for elem in self.weight_hyperparameters:
			 print('Weight list:',elem)
		self.ordered_weights = False
		self.bw_factor = bw_factor
		return
	# Commenting this out in the hope I don't need it any more?
	# def order_weights(self):
	#   assert False #Haven't implemented reweighting to mu=0,sigma=1 for this function yet.
	# 	for batch_i in range(self.N_batch):
	# 		MCMC_batch_i = self.list_of_data_samples[batch_i]
	# 		MCMC_ordered_batch_i = MCMC_batch_i[self.nonweight_hyperparameters]
	# 		for weight_parameter_list_0 in self.weight_hyperparameters:
	# 			weight_parameter_list_i = [elem for elem in weight_parameter_list_0 if 'weight' in elem]
	# 			# print('weight param i',weight_parameter_list_i)
	# 			N_comp_i = len(weight_parameter_list_i)
	# 			prefix_i = weight_parameter_list_i[0].split('_weight')[0]
	# 			weight_order_i = np.argsort(MCMC_batch_i[weight_parameter_list_i].median(axis=0)).to_numpy()
	# 			# print('Weight order i',weight_order_i)
	# 			# print('Batch:',batch_i,'Weight:',weight_parameter_list_i)
	# 			# print('Median:',MCMC_batch_i[weight_parameter_list_i].median(axis=0))
	# 			weight_order_i = np.array([[f'{prefix_i}_mu_{elem}',f'{prefix_i}_scale_{elem}',f'{prefix_i}_weights_{elem}'] for elem in np.arange(N_comp_i)[weight_order_i]]).flatten().tolist()
	# 			# print('Final weight order ii',weight_order_i)
	# 			# print(type(MCMC_batch_i),'columns',MCMC_batch_i.columns)
	# 			MCMC_batch_weight_columns_i = MCMC_batch_i[weight_order_i].to_numpy()
	# 			ordered_column_names = np.array([[f'{prefix_i}_mu_{elem}',f'{prefix_i}_scale_{elem}',f'{prefix_i}_weights_{elem}'] for elem in np.arange(N_comp_i)]).flatten().tolist()
	# 			MCMC_ordered_batch_i = pd.concat([MCMC_ordered_batch_i,
	# 											 pd.DataFrame(MCMC_batch_weight_columns_i,columns = ordered_column_names)],axis=1)
	# 		self.list_of_data_samples[batch_i] = MCMC_ordered_batch_i
	# 	self.ordered_weights = True
	# 	return self
	def find_kde_product_0(self,MCMC_db,N_batch,N_steps,list_of_weight_hyperparams,population_hyperparameters):
			'''
			list_of_weight_hyperparams: E.g. ['alpha'] to collectively account for ['alpha_mu_0','alpha_scale_0','alpha_weights_0','alpha_mu_1','alpha_scale_1','alpha_weights_1',...]
			population_hyperparameters: E.g. ['OM','Ode',...,'alpha_mu_0','alpha_scale_0','alpha_weights_0','alpha_mu_1','alpha_scale_1','alpha_weights_1',...]
			'''
			# assert self.ordered_weights #Must have ordered the hyperparameter weights first - Update - don't think I do as they're all degenerate?
			MCMC_db = [MCMC_db[b_iii][population_hyperparameters] for b_iii in range(N_batch)]
			print('list',list_of_weight_hyperparams)
			list_of_weight_components = {elem:len(MCMC_db[0].filter(like=f'{elem}_weights',axis=1).columns) for elem in list_of_weight_hyperparams}
			print('dict',list_of_weight_components)
			dict_of_list_of_weight_indx = {} #Where the weights are in the MCMC_db columns, for a specific hyperparameter
			for weight_hyperparam in list_of_weight_hyperparams:
				dict_of_list_of_weight_indx[weight_hyperparam] = [np.where(MCMC_db[0].columns==f'{weight_hyperparam}_weights_{comp_i}')[0][0] for comp_i in range(list_of_weight_components[weight_hyperparam])]
			# print(list_of_list_of_weight_indx)
			# print('Input',[MCMC_db[elem].T for elem in range(N_batch)])
			kde = {elem:gaussian_kde(MCMC_db[elem].T,bw_method=self.bw_factor) for elem in range(N_batch)}
			def gaussian_kde_product_log_prob(x):
				for hyperparam_indx in dict_of_list_of_weight_indx.keys():
					for comp_indx in dict_of_list_of_weight_indx[hyperparam_indx]:
						if x[comp_indx]<0 or x[comp_indx]>1: return -np.inf #Weights must be 0<=x<=1
				# if (x[2]>x[5]) or (x[5]>x[8]): return -np.inf #Not currently asserting weight ordering as not imposed in the MCMC?
				log_prob =  np.sum(np.array([kde[elem].logpdf(x) for elem in range(N_batch)]),axis=0)
				return log_prob
			n_walkers = 2*len(population_hyperparameters)+2
			sampler = EnsembleSampler(nwalkers=n_walkers,ndim=len(population_hyperparameters),log_prob_fn=gaussian_kde_product_log_prob)
			cur_state_0 = [] 
			#Adding one here, as the final weight isn't included the list to make sure one of the weights isn't determinstic in the MCMC.
			#Adding one, so the sum of all but the last weights sum up to less than one:
			cur_simplex = {weight_hyperparam:np.random.dirichlet([1]*(list_of_weight_components[weight_hyperparam]+1),size=(n_walkers,1)) for 
								weight_hyperparam in list_of_weight_hyperparams}
			cur_simplex_iter = {elem:0 for elem in list_of_weight_hyperparams}
			# Not currently putting weights in order
			for col_i in MCMC_db[0].columns:
				min_hyper_val = np.min([np.min(MCMC_db[b_ii][col_i]) for b_ii in range(N_batch)])
				max_hyper_val = np.max([np.max(MCMC_db[b_ii][col_i]) for b_ii in range(N_batch)])
				if 'weight' in col_i: 
					hyperparam_ii = col_i.split('_weights')[0]
					w_ii = cur_simplex[hyperparam_ii].T[cur_simplex_iter[hyperparam_ii]].T
					cur_state_0.append(w_ii);cur_simplex_iter[hyperparam_ii]+=1
				else: 
					cur_state_0.append(np.random.uniform(low=min_hyper_val,high=max_hyper_val,size=(n_walkers,1)))
			cur_state = np.concatenate(cur_state_0,axis=1)
			_ = sampler.run_mcmc(cur_state,N_steps,progress=True,skip_initial_state_check=True)
			return sampler,kde

from sklearn.neighbors import KernelDensity
import warnings
def find_kde_product_reweighting(MCMC_db,N_batch,N_steps,list_of_weight_hyperparams,population_hyperparameters,bounded_dict = {},
								 rescaling_dict=None,kernel='gaussian',bandwidth=0.1,p_value=2.0):
	'''
	list_of_weight_hyperparams: E.g. ['alpha'] to collectively account for ['alpha_mu_0','alpha_scale_0','alpha_weights_0','alpha_mu_1','alpha_scale_1','alpha_weights_1',...]
	population_hyperparameters: E.g. ['OM','Ode',...,'alpha_mu_0','alpha_scale_0','alpha_weights_0','alpha_mu_1','alpha_scale_1','alpha_weights_1',...]
	bounded_params: E.g. ['OM']: Parameters bounded between 0 and 1
	'''
	if rescaling_dict is None:
		rescaling_dict = {'mu':{elem:0 for elem in population_hyperparameters},
						  'std':{elem:1 for elem in population_hyperparameters}} #I.e. apply no prescaling unless set.
	print(rescaling_dict)
	# assert self.ordered_weights #Must have ordered the hyperparameter weights first - Update - don't think I do as they're all degenerate?
	MCMC_db = [MCMC_db[b_iii][population_hyperparameters] for b_iii in range(N_batch)]
	reweighted_MCMC_db = [(MCMC_db[b_iii]-rescaling_dict['mu'])/rescaling_dict['std'] for b_iii in range(N_batch)]
	print('list',list_of_weight_hyperparams)
	list_of_weight_components = {elem:len(reweighted_MCMC_db[0].filter(like=f'{elem}_weights',axis=1).columns) for elem in list_of_weight_hyperparams}
	print('dict',list_of_weight_components)
	dict_of_list_of_weight_indx = {} #Where the weights are in the MCMC_db columns, for a specific hyperparameter
	for weight_hyperparam in list_of_weight_hyperparams:
		dict_of_list_of_weight_indx[weight_hyperparam] = [np.where(reweighted_MCMC_db[0].columns==f'{weight_hyperparam}_weights_{comp_i}')[0][0] for comp_i in range(list_of_weight_components[weight_hyperparam])]
	bounded_params_indx = {elem:np.where(reweighted_MCMC_db[0].columns==elem)[0] for elem in bounded_dict.keys()}
	# print(list_of_list_of_weight_indx)
	# print('Input',[reweighted_MCMC_db[elem].T for elem in range(N_batch)])
	# print('Input',reweighted_MCMC_db[0].describe())
	# kde = {elem:gaussian_kde(reweighted_MCMC_db[elem].T,bw_method=1.0) for elem in range(N_batch)}
	# kde = {elem: KernelDensity(kernel='tophat',bandwidth=0.5).fit(reweighted_MCMC_db[elem]) for elem in range(N_batch)}
	# kde = {elem: NaiveKDE(kernel='gaussian', bw=0.1,norm=p_value).fit(data=reweighted_MCMC_db[elem].to_numpy()) for elem in range(N_batch)}
	#Perhaps best so far:
	# kde = {elem: TreeKDE(kernel='gaussian', bw=1.0,norm=p_value).fit(data=reweighted_MCMC_db[elem].to_numpy()) for elem in range(N_batch)}
	# kde = {elem: TreeKDE(kernel=kernel, bw=bandwidth,norm=p_value).fit(data=reweighted_MCMC_db[elem].to_numpy()) for elem in range(N_batch)}
	kde = {elem:GMM(n_components=1000).fit(X=reweighted_MCMC_db[elem])for elem in tqdm(range(N_batch))}

	def gaussian_kde_product_log_prob(x):
		for hyperparam_indx in dict_of_list_of_weight_indx.keys():
			for comp_indx in dict_of_list_of_weight_indx[hyperparam_indx]:
				hyper_param_iii = reweighted_MCMC_db[0].columns[comp_indx]
				if ((x[comp_indx]*rescaling_dict['std'][hyper_param_iii])+rescaling_dict['mu'][hyper_param_iii])<0 or \
				   ((x[comp_indx]*rescaling_dict['std'][hyper_param_iii])+rescaling_dict['mu'][hyper_param_iii])>1: return -np.inf #Weights must be 0<=x<=1
		for bound_ii in bounded_dict.keys():
			if ((x[bounded_params_indx[bound_ii]]*rescaling_dict['std'][bound_ii])+rescaling_dict['mu'][bound_ii])<bounded_dict[bound_ii][0] or\
				((x[bounded_params_indx[bound_ii]]*rescaling_dict['std'][bound_ii])+rescaling_dict['mu'][bound_ii])>bounded_dict[bound_ii][1]: return -np.inf #Must be 0<=x<=1
		# if (x[2]>x[5]) or (x[5]>x[8]): return -np.inf #Not currently asserting weight ordering as not imposed in the MCMC?
		# log_prob =  np.sum(np.array([kde[elem].score([x]) for elem in range(N_batch)]),axis=0)
		# log_prob = np.sum(np.array([kde[elem].logpdf(x) for elem in range(N_batch)]),axis=0)
		# log_prob = float(np.sum(np.array([np.log(kde[elem].evaluate(np.array([x]))) for elem in range(N_batch)]),axis=0))
		log_prob = float(np.sum(np.array([kde[elem].score([x]) for elem in range(N_batch)]),axis=0))
		if np.random.random()<0.001: print(log_prob)
		return log_prob
	n_walkers = 2*len(population_hyperparameters)+2
	sampler = EnsembleSampler(nwalkers=n_walkers,ndim=len(population_hyperparameters),log_prob_fn=gaussian_kde_product_log_prob)
	cur_state_0 = [] 
	#Adding one here, as the final weight isn't included the list to make sure one of the weights isn't determinstic in the MCMC.
	#Adding one, so the sum of all but the last weights sum up to less than one:
	cur_simplex = {weight_hyperparam:np.random.dirichlet([1]*(list_of_weight_components[weight_hyperparam]+1),size=(n_walkers,1)) for 
						weight_hyperparam in list_of_weight_hyperparams}
	cur_simplex_iter = {elem:0 for elem in list_of_weight_hyperparams}
	# Not currently putting weights in order
	for col_i in reweighted_MCMC_db[0].columns:
		min_hyper_val = np.min([np.min(reweighted_MCMC_db[b_ii][col_i]) for b_ii in range(N_batch)])
		max_hyper_val = np.max([np.max(reweighted_MCMC_db[b_ii][col_i]) for b_ii in range(N_batch)])
		if 'weight' in col_i: 
			hyperparam_ii = col_i.split('_weights')[0]
			w_ii = (cur_simplex[hyperparam_ii].T[cur_simplex_iter[hyperparam_ii]].T-rescaling_dict['mu'][col_i])/rescaling_dict['std'][col_i]
			cur_state_0.append(w_ii);cur_simplex_iter[hyperparam_ii]+=1
		else: 
			cur_state_0.append(np.random.uniform(low=min_hyper_val,high=max_hyper_val,size=(n_walkers,1)))
	cur_state = np.concatenate(cur_state_0,axis=1)
	with warnings.catch_warnings(action="ignore",category=UserWarning):
		_ = sampler.run_mcmc(cur_state,N_steps,progress=True,skip_initial_state_check=True)
	samples = {walker_i:pd.DataFrame(sampler.get_chain()[:,walker_i,:],columns=population_hyperparameters)*rescaling_dict['std']+rescaling_dict['mu']\
											for walker_i in range(sampler.get_chain().shape[1])}
	return samples,'Not returning KDE as it is applied to the reweighted samples so doesnt make sense on its own.'


