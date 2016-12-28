# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 12:54:42 2016

@author: ethan
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.integrate import quad
from scipy import interpolate

"""
this code runs assuming the following directory setup:
parent_dir/
    data/
        SCPUnion2.1_mu_vs_z.txt
    figures/
    scripts/
        this_script.py
"""

###############Functions###############

def H(z,OM,H0,w):
    if w==-1:
        return H0*np.sqrt(OM*(1+z)**3+(1-OM))
    else:
        return H0*np.sqrt(OM*(1+z)**3+(1-OM)*(1+z)**(3*(w+1)))

def dL(z,OM,H0,w):
    c=2.99792e5
    return (1+z)*quad(lambda x:c/H(x,OM,H0,w),0,z)[0]

def mu_w(z,OM,H0,w):
    return 5*np.log10(dL(z,OM,H0,w))+25

mu_w_vectorized = np.vectorize(mu_w) #allows z input to be an array

def mu_interp(z,OM,H0,w):
    z_spl = np.linspace(np.min(z),np.max(z),50)
    mu_spl = mu_w_vectorized(z_spl,OM,H0,w)
    tck = interpolate.splrep(z_spl, mu_spl)
    mu_int = interpolate.splev(z, tck)
    return mu_int
    

###############Data manipulation###############

z_obs,mu_obs,sig_mu_obs = np.loadtxt('../data/fakedata1.txt',
                                     usecols=[0,1,2],unpack=True)


###############MCMC points###############

om_nb,H0_nb,w_nb = np.loadtxt('../data/mcmc_fd1_nobias.txt',usecols=[1,2,3],unpack=True)


###############Scatter plot, bias plot and histogram###############

sns.set(style="white")#darkgrid, whitegrid, dark, white, ticks

plt.figure(num=None, figsize=(10, 8), dpi=80, facecolor='w', edgecolor='k')
plt.subplot(2,2,1)
plt.scatter(om_nb,H0_nb,s=4,c=[0.8,0,0],alpha=0.05)
plt.title(r'$\rm{MCMC\ Scatter}$',fontsize=15)#,y=1.02)
plt.ylabel(r'$H_0\ \rm[km/s/Mpc]$',fontsize=14)
plt.xlabel('$\Omega_M$',fontsize=14)
plt.xlim([0,0.5])

plt.subplot(2,2,2)
plt.scatter(om_nb,w_nb,s=4,c=[0.8,0,0],alpha=0.05)
plt.title(r'$\rm{MCMC\ Scatter}$',fontsize=15)#,y=1.02)
plt.ylabel(r'$w$',fontsize=14)
plt.xlabel('$\Omega_M$',fontsize=14)
plt.xlim([0,0.5])

z = np.linspace(10**-2,1.6,100)
plt.subplot(2,2,3)
plt.plot(z,mu_w_vectorized(z,0.31,67.74,-1),color=[0.8,0,0])
plt.plot(z,mu_w_vectorized(z,np.median(om_nb),np.median(H0_nb),np.median(w_nb)),color=[0,0,0.8])
plt.errorbar(z_obs,mu_obs,yerr=sig_mu_obs,marker=".",ms=6,fmt='.k',alpha=0.2,elinewidth=1)
plt.title(r'$\rm{Distance\ Modulus}$',fontsize=15)#,y=1.02)
plt.xlabel(r'$\rm{Redshift\ (z)}$',fontsize=14)
plt.ylabel(r'$\rm{\mu (z)\ (mag)}$',fontsize=14)
plt.legend([r'$\rm{Fiducial\ Model}$',r'$\rm{Best\ Fit}$',r'$\rm{Simulated\ Data}$'],
           loc=4,numpoints=1,markerscale=2)
plt.xlim([0,1.6])

plt.subplot(2,2,4)
plt.errorbar(z_obs,mu_obs-mu_w_vectorized(z_obs,0.31,67.74,-1),yerr=sig_mu_obs,
             marker=".",ms=8,fmt='.k',alpha=0.2,elinewidth=1)
plt.title(r'$\rm{Distance\ Modulus\ Residual}$',fontsize=15)#,y=1.02)
plt.xlabel(r'$\rm{Redshift\ (z)}$',fontsize=14)
plt.ylabel(r'$\rm{\Delta \mu (z)\ (mag)}$',fontsize=14)
plt.xlim([0,1.6])

plt.tight_layout()
plt.savefig('../figures/no_bias_fd1.png')
plt.close()