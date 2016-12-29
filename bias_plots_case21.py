# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 12:54:42 2016

@author: ethan
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random as rand
from scipy.integrate import quad
from scipy import interpolate

"""
this code runs assuming the following directory setup:
parent_dir/
    data/
        data_file.txt
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

sig_mu_obs = np.sqrt(sig_mu_obs**2 + 0.1**2) #add instrinsic scatter

z_obs0 = []
z_obs1 = []
z_obs2 = []
mu_obs1 = []
mu_obs2 = []
sig_mu_obs1 = []
sig_mu_obs2 = []

snpop = 1.0
shift = 0.01

"""
this loop shifts the z coordinate by a random amount given by the variable
'shift' with a random chance given by 'snpop'. it generates a contaminated
supernovae sample, as well as the split contaminant and non-contaminant 
samples.
z_obs0: contaminated sample
z_obs1: non-contaminants only
z_obs2: contaminants only"""

rand.seed(7)
for i in range(0,len(z_obs)):
    MC = rand.random()
    if MC > snpop:
        z_obs0.append(z_obs[i])
        z_obs1.append(z_obs[i])
        mu_obs1.append(mu_obs[i])
        sig_mu_obs1.append(sig_mu_obs[i])
    else:
        shifted_z_obs = z_obs[i]-(rand.gauss(0,shift*(1+z_obs[i])))
        while shifted_z_obs <= 0:
            shifted_z_obs = z_obs[i]-(rand.gauss(0,shift*(1+z_obs[i])))
        z_obs0.append(shifted_z_obs)
        z_obs2.append(shifted_z_obs)
        mu_obs2.append(mu_obs[i])
        sig_mu_obs2.append(np.sqrt(sig_mu_obs[i]**2 + (shift*(1+z_obs[i]))**2))

sig_mu_obs2 = np.array(sig_mu_obs2)


###############MCMC points###############

om_nb,H0_nb,w_nb = np.loadtxt('../data/mcmc_fd1_nobias.txt', usecols=[1,2,3],unpack=True)
om_bi,H0_bi,w_bi = np.loadtxt('../data/mcmc_fd1_case21.txt', usecols=[1,2,3],unpack=True)


###############Scatter plot, bias plot and histogram###############

sns.set(style="white")#darkgrid, whitegrid, dark, white, ticks

plt.figure(num=None, figsize=(10, 8), dpi=80, facecolor='w', edgecolor='k')
plt.subplot(2,2,1)
plt.scatter(.3,70,s=4,c=[0.8,0,0],alpha=1)
plt.scatter(.3,70,s=4,c=[0,0.8,0.8],alpha=1)
plt.scatter(om_nb,H0_nb,s=4,c=[0.8,0,0],alpha=0.05)
plt.scatter(om_bi,H0_bi,s=4,c=[0,0.8,0.8],alpha=0.05)
plt.title(r'$\rm{MCMC\ Scatter}$',fontsize=15)#,y=1.02)
plt.ylabel(r'$H_0\ \rm[km/s/Mpc]$',fontsize=14)
plt.xlabel('$\Omega_M$',fontsize=14)
plt.legend([r'$\rm{Unbiased}$',r'$\rm{Biased}$'],loc=4,markerscale=2)
plt.xlim([0,0.5])

plt.subplot(2,2,2)
plt.scatter(.3,-1,s=4,c=[0.8,0,0],alpha=1)
plt.scatter(.3,-1,s=4,c=[0,0.8,0.8],alpha=1)
plt.scatter(om_nb,w_nb,s=4,c=[0.8,0,0],alpha=0.05)
plt.scatter(om_bi,w_bi,s=4,c=[0,0.8,0.8],alpha=0.05)
plt.title(r'$\rm{MCMC\ Scatter}$',fontsize=15)#,y=1.02)
plt.ylabel(r'$w$',fontsize=14)
plt.xlabel('$\Omega_M$',fontsize=14)
plt.legend([r'$\rm{Unbiased}$',r'$\rm{Biased}$'],loc=4,markerscale=2)
plt.xlim([0,0.5])

z = np.linspace(10**-2,1.6,100)
plt.subplot(2,2,3)
plt.plot(z,mu_w_vectorized(z,0.31,67.74,-1),color=[0.8,0,0])
plt.plot(z,mu_w_vectorized(z,np.median(om_bi),np.median(H0_bi),np.median(w_bi)),color=[0,0,0.8])
plt.errorbar(z_obs0,mu_obs,yerr=sig_mu_obs2,marker=".",ms=6,fmt='.k',alpha=0.2,elinewidth=1)
plt.title(r'$\rm{Distance\ Modulus}$',fontsize=15)#,y=1.02)
plt.xlabel(r'$\rm{Redshift\ (z)}$',fontsize=14)
plt.ylabel(r'$\rm{\mu (z)\ (mag)}$',fontsize=14)
plt.legend([r'$\rm{Fiducial\ Model}$',r'$\rm{Best\ Fit}$',r'$\rm{Simulated\ Data}$'],
           loc=4,numpoints=1,markerscale=2)
plt.xlim([0,1.6])

plt.subplot(2,2,4)
plt.errorbar(z_obs0,mu_obs2-mu_w_vectorized(z_obs0,0.31,67.74,-1),yerr=sig_mu_obs2,
             marker="v",ms=6,fmt='.r',alpha=0.2,elinewidth=1)
plt.errorbar(z_obs,mu_obs-mu_w_vectorized(z_obs,0.31,67.74,-1),yerr=sig_mu_obs,
             marker=".",ms=8,fmt='.k',alpha=0.2,elinewidth=1)
plt.title(r'$\rm{Distance\ Modulus\ Residual}$',fontsize=15)#,y=1.02)
plt.xlabel(r'$\rm{Redshift\ (z)}$',fontsize=14)
plt.ylabel(r'$\rm{\Delta \mu (z)\ (mag)}$',fontsize=14)
plt.legend([r'$\rm{Unbiased}$',r'$\rm{Biased}$'],loc=4,markerscale=2)
plt.xlim([0,1.6])

plt.tight_layout()
plt.savefig('../figures/case21_fd1.png')
plt.close()