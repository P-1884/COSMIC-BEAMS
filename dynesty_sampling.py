from sklearn.mixture import GaussianMixture as GMM
import numpy as np
import matplotlib.pyplot as pl
import pandas as pd
import dynesty
from pathos.pools import ProcessPool as Pool
from multiprocessing import cpu_count
from KDEpy import TreeKDE
from sklearn.neighbors import KernelDensity
print('Loading data')
db = pd.read_csv('./test_data_for_download.csv')
db.drop('Unnamed: 0', axis=1, inplace=True)

db = db[['OM','Ode','w','wa',
    'alpha_scale_0','alpha_scale_1','alpha_scale_2',
    'alpha_mu_0','alpha_mu_1','alpha_mu_2',
    'alpha_weights_0','alpha_weights_1',
    's_c','s_m',
    'scale_c','scale_m']]

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
# type='GMM'
# G1000 = GMM(n_components=N_comp).fit(db)
# kde = G1000

def gaussian_kde_product_log_prob(x):
    # x = pd.DataFrame(x, index=db.columns.tolist()).T
    # log_prob = kde.score(x)
    log_prob = kde.score([x])
    # log_prob = np.log(kde.evaluate(np.array([x]))).item()
    return log_prob#.flatten()

def prior_transform(x):
    return x

print('Starting Pool')
N_threads = cpu_count()
thread = Pool(ncpus=N_threads)

print(f'Defining Sampler with {N_threads} threads')
sampler = dynesty.DynamicNestedSampler(gaussian_kde_product_log_prob, prior_transform,
                            len(db.columns),bound='none', pool = thread,
                                      queue_size=N_threads)

print('Running Sampler')
sampler.run_nested()
sampler.run_nested(print_progress=True)
#
print('Getting Results')
sresults = sampler.results
weights = sresults.importance_weights()
samples_equal = sresults.samples_equal()

# import corner
# fig = pl.figure(figsize=(30,30))
# corner.corner(samples_equal,labels=db.columns.tolist(),fig=fig,plot_datapoints=False,
# hist_kwargs={'density':True},
# hist2d_kwargs={'label':'_nolegend_'},color='k',bins=50)

# corner.corner(G1000.sample(len(db))[0],labels=db.columns.tolist(),fig=fig,plot_datapoints=False,
# hist_kwargs={'density':True},
# hist2d_kwargs={'label':'_nolegend_'},color='green',bins=50)

# corner.corner(db,labels=db.columns.tolist(),fig=fig,plot_datapoints=False,
# hist_kwargs={'density':True},
# hist2d_kwargs={'label':'_nolegend_'},color='darkred',bins=50)
# pl.tight_layout()
# pl.show()

print('Saving Samples')
pd.DataFrame(samples_equal).to_csv(f'/mnt/extraspace/hollowayp/zBEAMS_data/dynesty_samples/dynesty_samples_{type}_{N_comp}.csv')