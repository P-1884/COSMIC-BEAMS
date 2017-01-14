# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 12:54:42 2016

@author: ethan
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import gridspec
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

def dmudz(z,OM,H0,w):
    c=2.99792e5
    return 5/np.log(10)*(1/(1+z)+(1+z)*c/(dL(z,OM,H0,w)*H(z,OM,H0,w)))

dmu_vectorized = np.vectorize(dmudz)

def dmu_interp(z,OM,H0,w):
    z_spl = np.linspace(np.min(z),np.max(z),50)
    dmu_spl = dmu_vectorized(z_spl,OM,H0,w)
    tck = interpolate.splrep(z_spl, dmu_spl)
    dmu_int = interpolate.splev(z, tck)
    return dmu_int
    

###############Data manipulation###############

z_obs,mu_obs,sig_mu_obs = np.loadtxt('../data/fakedata1_er.txt',
                                     usecols=[0,2,3],unpack=True)

erm = 'c' #'a' or 'b' or 'c' (how we deal with error)

z_obs0 = []
z_obs1 = []
z_obs2 = []
mu_obs1 = []
mu_obs2 = []
sig_mu_obs1 = []
sig_mu_obs2 = []

snpop = 1.0
shift = 0.04

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
        while shifted_z_obs <= 0.01:
            shifted_z_obs = z_obs[i]-(rand.gauss(0,shift*(1+z_obs[i])))
        z_obs0.append(shifted_z_obs)
        z_obs2.append(shifted_z_obs)
        mu_obs2.append(mu_obs[i])
        dmu = dmudz(z_obs[i],0.31,67.74,-1)
        if erm == 'a' or erm =='c':
            sig_mu_obs2.append(sig_mu_obs[i])#a/c
        elif erm =='b':
            sig_mu_obs2.append(np.sqrt(sig_mu_obs[i]**2 + (dmu*shift*(1+z_obs[i]))**2))#b

sig_mu_obs2 = np.array(sig_mu_obs2)
z_obs0 = np.array(z_obs0)


###############MCMC points###############

nb_mcmc_file = '../data/bias_chain_fd1_nobias_er.txt'
bi_mcmc_file = '../data/bias_chain_fd1_case2.1_'+erm+'_er.txt'

om_nb,H0_nb,w_nb = np.loadtxt(nb_mcmc_file, usecols=[0,1,2],unpack=True)
om_bi,H0_bi,w_bi = np.loadtxt(bi_mcmc_file, usecols=[0,1,2],unpack=True)


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
plt.xlim([0,0.7])

plt.subplot(2,2,2)
plt.scatter(.3,-1,s=4,c=[0.8,0,0],alpha=1)
plt.scatter(.3,-1,s=4,c=[0,0.8,0.8],alpha=1)
plt.scatter(om_nb,w_nb,s=4,c=[0.8,0,0],alpha=0.05)
plt.scatter(om_bi,w_bi,s=4,c=[0,0.8,0.8],alpha=0.05)
plt.title(r'$\rm{MCMC\ Scatter}$',fontsize=15)#,y=1.02)
plt.ylabel(r'$w$',fontsize=14)
plt.xlabel('$\Omega_M$',fontsize=14)
plt.legend([r'$\rm{Unbiased}$',r'$\rm{Biased}$'],loc=4,markerscale=2)
plt.xlim([0,0.7])

z = np.linspace(10**-2,1.2,100)
plt.subplot(2,2,3)
plt.plot(z,mu_w_vectorized(z,0.31,67.74,-1),color=[0.8,0,0])
plt.plot(z,mu_w_vectorized(z,np.median(om_bi),np.median(H0_bi),np.median(w_bi)),color=[0,0,0.8])
plt.errorbar(z_obs0,mu_obs,yerr=sig_mu_obs2,marker=".",ms=6,fmt='.k',alpha=0.2,elinewidth=1)
plt.title(r'$\rm{Distance\ Modulus}$',fontsize=15)#,y=1.02)
plt.xlabel(r'$\rm{Redshift\ (z)}$',fontsize=14)
plt.ylabel(r'$\rm{\mu (z)\ (mag)}$',fontsize=14)
plt.legend([r'$\rm{Fiducial\ Model}$',r'$\rm{Best\ Fit}$',r'$\rm{Simulated\ Data}$'],
           loc=4,numpoints=1,markerscale=2)
plt.xlim([0,1.2])

plt.subplot(2,2,4)
plt.errorbar(z_obs0,mu_obs2-mu_w_vectorized(z_obs0,0.31,67.74,-1),yerr=sig_mu_obs2,
             marker="v",ms=6,fmt='.r',alpha=0.2,elinewidth=1)
plt.errorbar(z_obs,mu_obs-mu_w_vectorized(z_obs,0.31,67.74,-1),yerr=sig_mu_obs,
             marker=".",ms=8,fmt='.k',alpha=0.2,elinewidth=1)
plt.title(r'$\rm{Distance\ Modulus\ Residual}$',fontsize=15)#,y=1.02)
plt.xlabel(r'$\rm{Redshift\ (z)}$',fontsize=14)
plt.ylabel(r'$\rm{\Delta \mu (z)\ (mag)}$',fontsize=14)
plt.legend([r'$\rm{Biased}$',r'$\rm{Unbiased}$'],loc=4,markerscale=2)
plt.xlim([0,1.2])

plt.tight_layout()
plt.savefig('../figures/bias_subplots_fd1_case2.1_'+erm+'_er.png')
plt.close()

#  Code to plot a contour from an MCMC chain

#Finds the 95% and 68% confidence intervals, given a 2d histogram of the likelihood
def findconfidence(H):
    H2 = H.ravel()
    H2 = np.sort(H2)
    
    #Cut out the very low end
    #H2 = H2[H2>100]

    #Loop through this flattened array until we find the value in the bin which contains 95% of the points
    tot = sum(H2)
    tot95=0
    tot68=0

    #Changed this to 68% and 30% C.I
    for i in range(len(H2)):
        tot95 += H2[i]
        if tot95 >= 0.05*tot:
            N95 = H2[i]
            #print i
            break

    for i in range(len(H2)):
        tot68 += H2[i]
        if tot68>=0.32*tot:
            N68 = H2[i]
            break   
    return max(H2),N95,N68


def contour(chain,p,**kwargs):
    binsize=50
    H, xedges, yedges = np.histogram2d(chain[:,p[0]],chain[:,p[1]], bins=(binsize,binsize))
    
    x=[]
    y=[]
    z=[]
    for i in range(len(xedges[:-1])):
        for j in range(len(yedges[:-1])):
            x.append(xedges[:-1][i])
            y.append(yedges[:-1][j])
            z.append(H[i, j])

    if 'smooth' in kwargs:
        SMOOTH=True
        smth=kwargs['smooth']
        if smth==0:
            SMOOTH=False
    else:
        SMOOTH=True
        smth=10e5
    if SMOOTH:
        sz=50
        spl = interpolate.bisplrep(x, y, z,  s=smth)
        X = np.linspace(min(xedges[:-1]), max(xedges[:-1]), sz)
        Y = np.linspace(min(yedges[:-1]), max(yedges[:-1]), sz)
        Z = interpolate.bisplev(X, Y, spl)
    else:
        X=xedges[:-1]
        Y=yedges[:-1]
        Z=H
    
    #I think this is the weird thing I have to do to make the contours work properly
    X1=np.zeros([len(X), len(X)])
    Y1=np.zeros([len(X), len(X)])
    for i in range(len(X)):
        X1[ :, i]=X
        Y1[i, :]=Y
    X=X1
    Y=Y1
    
    N100,N95,N68 = findconfidence(Z)
    
    return N100,N95,N68,X,Y,Z
    
    
c_nb = np.loadtxt(nb_mcmc_file,usecols=[0,2])
c_bi = np.loadtxt(bi_mcmc_file,usecols=[0,2])
N100_nb,N95_nb,N68_nb,X_nb,Y_nb,Z_nb = contour(c_nb,[0,1], labels=['1', '2'],line=False)
N100_bi,N95_bi,N68_bi,X_bi,Y_bi,Z_bi = contour(c_bi,[0,1], labels=['1', '2'],line=False)
col0 = ('#a3c0f6','#0057f6')
col1 = ('#fc9272','#de2d26')

sns.set(style="white")#darkgrid, whitegrid, dark, white, ticks

plt.figure(num=None, figsize=(10, 8), dpi=80, facecolor='w', edgecolor='k')
gs = gridspec.GridSpec(2, 2, width_ratios=[3,1], height_ratios=[1,3])
ax=plt.subplot(gs[1,0])
cs0 = plt.contourf(X_nb,Y_nb,Z_nb,levels=[N95_nb,N68_nb,N100_nb],colors=col0,alpha=0.8)
cs1 = plt.contourf(X_bi,Y_bi,Z_bi,levels=[N95_bi,N68_bi,N100_bi],colors=col1,alpha=0.8)
cs2 = plt.scatter(0.31,-1,s=16,marker='x',c='k',label=r'$\rm{Fiducial\ Model}$')
plt.title(r'$\rm{MCMC\ Contours\ Case\ 2.1c}$',fontsize=15)#,y=1.02)
plt.ylabel(r'$w$',fontsize=14)
plt.xlabel('$\Omega_M$',fontsize=14)
plt.ylim([-2.5,0])

patch1 = mpatches.Patch(color=col1[0], label=r'$\rm{Biased\ 95\%\ CL}$')
patch2 = mpatches.Patch(color=col1[1], label=r'$\rm{Biased\ 68\%\ CL}$')
patch3 = mpatches.Patch(color=col0[0], label=r'$\rm{Unbiased\ 95\%\ CL}$')
patch4 = mpatches.Patch(color=col0[1], label=r'$\rm{Unbiased\ 68\%\ CL}$')
plt.legend(handles=[cs2,patch1,patch2,patch3,patch4],loc=4)

plt.subplot(gs[0,0],sharex=ax)
plt.hist(om_nb,bins=50,normed=True,color=col0[0],alpha=0.6)
plt.hist(om_bi,bins=50,normed=True,color=col1[0],alpha=0.6)
plt.xlim([0,0.7])
plt.yticks([])

plt.subplot(gs[1,1],sharey=ax)
plt.hist(w_nb,bins=50,normed=True,orientation='horizontal',color=col0[0],alpha=0.6)
plt.hist(w_bi,bins=50,normed=True,orientation='horizontal',color=col1[0],alpha=0.6)
plt.xticks([])

plt.savefig('../figures/bias_contour_fd1_case2.1_'+erm+'_er.png')
plt.close()
