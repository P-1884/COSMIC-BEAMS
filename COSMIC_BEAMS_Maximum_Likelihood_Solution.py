print('Loading Packages')
import os
os.environ['JAX_PLATFORMS'] = 'cpu'
# os.environ['JAX_ENABLE_X64']='True'
# import jax
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
import glob
import sys
import jax
import os
try:importlib.reload(sys.modules['mcmcfunctions_SL'])
except Exception as ex: print(f'Cannot reload: {ex}')

from mcmcfunctions_SL import mcmc_SL

Om_fid = 0.3;Ode_fid = 0.7;H0_fid = 70;w_fid = -1.0;wa_fid=0

cosmo_type = 'wCDM'
from mcmcfunctions_SL_JAX import j_likelihood_SL,run_MCMC

db_100k = pd.read_csv('./databases/real_paltas_population_TP_100000_FP_0_Spec_10000_P_1.0.csv')

from cosmology_JAX_flat import j_r_SL as j_r_SL_flat
from cosmology_JAX import j_r_SL

class likelihood_gradient:
    def __init__(self,Ok_val,db_test,no_parent):
        self.Ok_val = Ok_val
        self.db_test = db_test
        self.no_parent=no_parent
    #zL_obs,zS_obs,sigma_zL_obs,sigma_zS_obs,r_obs,sigma_r_obs
        self.likelihood_dict_default = {'H0':70,'OM':0.3, 'Ok':self.Ok_val, # The default is Om=0 and Ok = 0. This is so when one of them is updated
                               # in likelihood_dict below, it doesn't require Ode to be >1. Otherwise if I let Om or Ok vary between
                               # [0,1] and the other (Ok or Om) is non-zero by default, it would require Ode to be >1.
                                'w':-1,'wa':0,
                                'sigma_r_obs':jnp.array([0.1]),
                                # 'zL':jnp.linspace(0.8,1.2,101),'zS':jnp.linspace(1.8,2.2,101),
                                # 'zL_obs':jnp.linspace(0.8,1.2,101),'zS_obs':jnp.linspace(1.8,2.2,101),
                                # 'zL_sigma':0.001,'zS_sigma':0.001,
                                # 'mu_zL_g_L':1,'mu_zS_g_L':2,
                                'zL':self.db_test['zL_true'].to_numpy(),'zS':self.db_test['zS_true'].to_numpy(),
                                # 'zL_obs':np.random.normal(loc=self.db_test['zL_true'],scale=self.db_test['sigma_zL_obs']),
                                # 'zS_obs':np.random.normal(loc=self.db_test['zS_true'],scale=self.db_test['sigma_zS_obs']),
                                # 'zL_obs':self.db_test['zL_obs'].to_numpy(),'zS_obs':self.db_test['zS_obs'].to_numpy(),
                                # 'zL_obs':self.db_test['zL_obs'].to_numpy(),'zS_obs':self.db_test['zS_obs'].to_numpy(),
                                'zL_obs':self.db_test['zL_true'].to_numpy(),'zS_obs':self.db_test['zS_true'].to_numpy(),
                                'zL_sigma':self.db_test['sigma_zL_obs'].to_numpy(),'zS_sigma':self.db_test['sigma_zS_obs'].to_numpy(),
                                'mu_zL_g_L':np.mean(self.db_test['zL_true']),'mu_zS_g_L':np.mean(self.db_test['zS_true']),
                                'sigma_01_g_L':np.nan,'sigma_01_g_L':np.nan}
        self.likelihood_dict_default['sigma_zL_g_L'] = jnp.std(self.likelihood_dict_default['zL'])
        self.likelihood_dict_default['sigma_zS_g_L'] = jnp.std(self.likelihood_dict_default['zS'])
        self.likelihood_dict_default['r_obs'] = j_r_SL_flat(self.likelihood_dict_default['zL'],
                                                    self.likelihood_dict_default['zS'],
                                               jc.Cosmology(
                                               Omega_c=self.likelihood_dict_default['OM'], 
                                               h=self.likelihood_dict_default['H0']/100,
                                               Omega_k=self.likelihood_dict_default['Ok'],
                                               w0=self.likelihood_dict_default['w'],
                                               Omega_b=0, wa=self.likelihood_dict_default['wa'], 
                                               sigma8=0.8, n_s=0.96)).copy()
    # print('Means',np.mean(self.likelihood_dict_default['zL']),np.mean(self.likelihood_dict_default['zS']))
    def simplified_likelihood(self,zL,zS,OM):
        likelihood_dict = self.likelihood_dict_default.copy()
        likelihood_dict['zL'] = zL
        likelihood_dict['zS'] = zS
        likelihood_dict['r'] = j_r_SL_flat(zL,zS,
                                               jc.Cosmology(
                                               Omega_c=OM, 
                                               h=self.likelihood_dict_default['H0']/100,
                                               Omega_k=self.likelihood_dict_default['Ok'],
                                               w0=self.likelihood_dict_default['w'],
                                               Omega_b=0, 
                                               wa=self.likelihood_dict_default['wa'], 
                                               sigma8=0.8, n_s=0.96))
        # print((likelihood_dict['r_obs']==likelihood_dict['r']).all())
        return j_likelihood_SL(np.nan*np.ones(len(self.db_test)),jnp.array([np.nan]),np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,
            cosmo_type='FlatwCDM',
            photometric=True,
            contaminated=False,
            H0=likelihood_dict['H0'],
            key=PRNGKey(0),
            likelihood_check = True,
            likelihood_dict = likelihood_dict,
            batch_bool=False,
            spec_indx = np.where(self.db_test['spec'])[0],
            no_parent=self.no_parent
            )
    def find_gradient(self):
        l_dict = {'zL':self.likelihood_dict_default['zL'],
                'zL_obs':self.likelihood_dict_default['zL_obs'],
                'zS':self.likelihood_dict_default['zS'],
                'zS_obs':self.likelihood_dict_default['zS_obs'],
                'OM':self.likelihood_dict_default['OM']}
        print('Total likelihood:',jnp.mean(abs(self.simplified_likelihood(l_dict['zL'],l_dict['zS'],l_dict['OM']))))
        print('WRT zL',jnp.mean(abs(jax.grad(self.simplified_likelihood,argnums=0)(l_dict['zL'],l_dict['zS'],l_dict['OM']))))
        print('WRT zS',jnp.mean(abs(jax.grad(self.simplified_likelihood,argnums=1)(l_dict['zL'],l_dict['zS'],l_dict['OM']))))
        print('WRT COSMO',jnp.mean(abs(jax.grad(self.simplified_likelihood,argnums=2)(l_dict['zL'],l_dict['zS'],l_dict['OM']))))
    def find_max_likelihood(self):
        def optimal_zL(zL):
            likelihood_dict = self.likelihood_dict_default.copy()
            likelihood_dict['zL'] = zL
            likelihood_dict['r'] = j_r_SL_flat(zL,self.likelihood_dict_default['zS'],
                                                jc.Cosmology(
                                                Omega_c=self.likelihood_dict_default['OM'], 
                                                h=self.likelihood_dict_default['H0']/100,
                                                Omega_k=self.likelihood_dict_default['Ok'],
                                                w0=self.likelihood_dict_default['w'],
                                                Omega_b=0, 
                                                wa=self.likelihood_dict_default['wa'], 
                                                sigma8=0.8, n_s=0.96))
            L =  j_likelihood_SL(np.nan*np.ones(len(self.db_test)),jnp.array([np.nan]),np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,
                cosmo_type='FlatwCDM',
                photometric=True,contaminated=False,
                H0=likelihood_dict['H0'],key=PRNGKey(0),
                likelihood_check = True,likelihood_dict = likelihood_dict,
                batch_bool=False,
                no_parent=self.no_parent)
            return L
        def optimal_OM(OM):
            likelihood_dict = self.likelihood_dict_default.copy()
            likelihood_dict['OM']=OM
            likelihood_dict['r'] = j_r_SL_flat(self.likelihood_dict_default['zL'],
                                               self.likelihood_dict_default['zS'],
                                                jc.Cosmology(
                                                Omega_c=OM, 
                                                h=self.likelihood_dict_default['H0']/100,
                                                Omega_k=self.likelihood_dict_default['Ok'],
                                                w0=self.likelihood_dict_default['w'],
                                                Omega_b=0, 
                                                wa=self.likelihood_dict_default['wa'], 
                                                sigma8=0.8, n_s=0.96))
            L =  j_likelihood_SL(np.nan*np.ones(len(self.db_test)),jnp.array([np.nan]),np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,
                cosmo_type='wCDM',
                photometric=True,contaminated=False,
                H0=likelihood_dict['H0'],key=PRNGKey(0),
                likelihood_check = True,
                likelihood_dict = likelihood_dict,
                batch_bool=False,
                no_parent=self.no_parent)
            # jax.debug.print('L {L}',L=L)
            return L
        def optimal_combo(params,N_sys):
            likelihood_dict = self.likelihood_dict_default.copy()
            self.likelihood_dict_default['r_obs'] = self.db_test['r_obs_contam'].to_numpy()
            likelihood_dict['zL'] = params[0:N_sys]
            likelihood_dict['zS'] = params[N_sys:2*N_sys]
            likelihood_dict['OM'] = params[2*N_sys]
            likelihood_dict['r'] = j_r_SL_flat(likelihood_dict['zL'],
                                               likelihood_dict['zS'],
                                                jc.Cosmology(
                                                Omega_c=likelihood_dict['OM'], 
                                                h=self.likelihood_dict_default['H0']/100,
                                                Omega_k=self.likelihood_dict_default['Ok'],
                                                w0=self.likelihood_dict_default['w'],
                                                Omega_b=0, 
                                                wa=self.likelihood_dict_default['wa'], 
                                                sigma8=0.8, n_s=0.96))
            L =  j_likelihood_SL(np.nan*np.ones(len(self.db_test)),jnp.array([np.nan]),np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,
                cosmo_type='FlatwCDM',
                photometric=True,contaminated=False,
                H0=likelihood_dict['H0'],key=PRNGKey(0),
                likelihood_check = True,likelihood_dict = likelihood_dict,
                batch_bool=False,
                no_parent=self.no_parent)
            return L
        import jax.numpy as jnp
        from jax import grad, jit, vmap
        from jax import random
        from jax import jacfwd, jacrev
        from jax.numpy import linalg
        from numpy import nanargmin,nanargmax 
        def paraboloid(x): return -optimal_combo(x,len(self.db_test)) #Want to maximise not minimise the likelihood
        # def paraboloid(x): return -optimal_zL(x) #Want to maximise not minimise the likelihood
        # def paraboloid(x): return -optimal_OM(x) #Want to maximise not minimise the likelihood
        minfunc = vmap(paraboloid)
        J = jacfwd(paraboloid)
        def minJacobian(x): 
            jax.debug.print('Gradients: Median: {a}, Mean: {b}',a=jnp.median(abs(J(x))),b=jnp.mean(abs(J(x))))
            # return x - 0.0001*J(x) #Good value for redshifts
            # return x - 0.001*J(x)  #Good value for cosmology 
            return x - 0.000001*J(x)  #Good value for combo 
        # domain = np.random.normal(loc=self.db_test['zL_obs'],scale=self.db_test['sigma_zL_obs'],
                                #   size=(50,len(self.db_test)))
        domain = np.array([self.db_test['zL_true'].to_numpy().tolist()+self.db_test['zS_true'].to_numpy().tolist() + [0.3]])
        # domain = jnp.array([0.6])#np.random.uniform(0,1,size=(1,))
        # print(f'Starting point {domain}')
        inferred_value = []
        vfuncHS = vmap(minJacobian)
        for epoch in tqdm(range(5)):
            domain = vfuncHS(domain)
            inferred_value.append(jnp.array(domain[jnp.argmin(minfunc(domain))]))
        minimums = minfunc(domain)
        arglist = jnp.argmin(minimums)
        argmin = jnp.array(domain[arglist])
        minimum = minimums[arglist] 
        def plot_inference(inferred_array,N_sys):
            inferred_array=np.array(inferred_array)
            fig,ax = pl.subplots(1,3,figsize=(15,5))
            ax[0].plot(inferred_array[:,:N_sys]-self.db_test['zL_true'].to_numpy().T,alpha=0.8)
            ax[1].plot(inferred_array[:,N_sys:2*N_sys]-self.db_test['zS_true'].to_numpy().T,alpha=0.8)
            ax[2].plot(inferred_array[:,2*N_sys],alpha=0.8)
            for p_i in range(3): ax[p_i].set_xlabel('Iteration',fontsize=15)
            ax[0].set_ylabel('Residual',fontsize=15)
            ax[1].set_ylabel('Residual',fontsize=15)
            ax[2].set_ylabel('Inferred Value',fontsize=15)
            for p_i in range(3): ax[p_i].set_title(['Lens Redshift','Source Redshift','OM'][p_i],fontsize=18)
            pl.tight_layout()
            for fmt in ['.pdf','.png']: pl.savefig(f'./Maximum_Likelihood_COSMIC_BEAMS_Solution{fmt}',bbox_inches='tight')
            pl.close()
            # pl.show()
        def plot_single_inference(inferred_value):
            pl.plot(np.array(inferred_value)-self.db_test['zL_true'].to_numpy().T)
            # pl.plot(np.array(inferred_value))
            pl.xlabel('Iteration',fontsize=15)
            # pl.ylabel('Inferred Value',fontsize=15)
            pl.ylabel('Residual',fontsize=15)
            pl.title('Gradient Ascent of Likelihood Function',fontsize=21)
            # pl.ylim(-5,5)
            pl.tight_layout()
            pl.show()
        plot_inference(inferred_value,len(self.db_test))
        print('LENS REDSHIFT')
        print(f'Mean absolute difference to true redshifts {np.round(np.mean(abs(argmin[:len(self.db_test)]-self.db_test["zL_true"])),5)}')
        print(f'Median absolute difference to true redshifts {np.round(np.median(abs(argmin[:len(self.db_test)]-self.db_test["zL_true"])),5)}')
        print('SOURCE REDSHIFT')
        print(f'Mean absolute difference to true redshifts {np.round(np.mean(abs(argmin[len(self.db_test):2*len(self.db_test)]-self.db_test["zS_true"])),5)}')
        print(f'Median absolute difference to true redshifts {np.round(np.median(abs(argmin[len(self.db_test):2*len(self.db_test)]-self.db_test["zS_true"])),5)}')
        print("The Max Log-likelihood is",minimum)#," the arg min is:",np.round(argmin,2))
        # print('Shape of output',np.shape(np.array(inferred_value)))

import warnings
db_samp = db_100k.sample(10000)
db_samp = db_samp[db_samp['spec']==0].reset_index(drop=True)
db_samp = db_samp.loc[np.random.choice(np.arange(len(db_samp)),size=100,replace=False)].reset_index(drop=True)
with warnings.catch_warnings(action="ignore"):
    likelihood_gradient(Ok_val=1e-10,db_test=db_samp,no_parent=True).find_max_likelihood()
# print(f'Median absolute difference between observations and truth {np.round(np.median(abs(db_samp["zL_obs"]-db_samp["zL_true"])),5)}')
# print(str(np.round(db_samp['zL_true'].to_numpy(),2)))
# print(str(np.round(db_samp['zL_obs'].to_numpy(),2)))
