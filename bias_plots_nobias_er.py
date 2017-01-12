# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 12:54:42 2016

@author: ethan
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
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

z_obs,mu_obs,sig_mu_obs = np.loadtxt('../data/fakedata1_er.txt',
                                     usecols=[0,2,3],unpack=True)


###############MCMC points###############

om_nb,H0_nb,w_nb = np.loadtxt('../data/bias_chain_fd1_nobias_er.txt',usecols=[0,1,2],unpack=True)


###############Scatter plot, bias plot and histogram###############

sns.set(style="white")#darkgrid, whitegrid, dark, white, ticks

plt.figure(num=None, figsize=(10, 8), dpi=80, facecolor='w', edgecolor='k')
plt.subplot(2,2,1)
plt.scatter(om_nb,H0_nb,s=4,c=[0.8,0,0],alpha=0.05)
plt.title(r'$\rm{MCMC\ Scatter}$',fontsize=15)#,y=1.02)
plt.ylabel(r'$H_0\ \rm[km/s/Mpc]$',fontsize=14)
plt.xlabel('$\Omega_M$',fontsize=14)
plt.xlim([0,0.7])

plt.subplot(2,2,2)
plt.scatter(om_nb,w_nb,s=4,c=[0.8,0,0],alpha=0.05)
plt.title(r'$\rm{MCMC\ Scatter}$',fontsize=15)#,y=1.02)
plt.ylabel(r'$w$',fontsize=14)
plt.xlabel('$\Omega_M$',fontsize=14)
plt.xlim([0,0.7])

z = np.linspace(10**-2,1.2,100)
plt.subplot(2,2,3)
plt.plot(z,mu_w_vectorized(z,0.31,67.74,-1),color=[0.8,0,0])
plt.plot(z,mu_w_vectorized(z,np.median(om_nb),np.median(H0_nb),np.median(w_nb)),color=[0,0,0.8])
plt.errorbar(z_obs,mu_obs,yerr=sig_mu_obs,marker=".",ms=6,fmt='.k',alpha=0.2,elinewidth=1)
plt.title(r'$\rm{Distance\ Modulus}$',fontsize=15)#,y=1.02)
plt.xlabel(r'$\rm{Redshift\ (z)}$',fontsize=14)
plt.ylabel(r'$\rm{\mu (z)\ (mag)}$',fontsize=14)
plt.legend([r'$\rm{Fiducial\ Model}$',r'$\rm{Best\ Fit}$',r'$\rm{Simulated\ Data}$'],
           loc=4,numpoints=1,markerscale=2)
plt.xlim([0,1.2])

plt.subplot(2,2,4)
plt.errorbar(z_obs,mu_obs-mu_w_vectorized(z_obs,0.31,67.74,-1),yerr=sig_mu_obs,
             marker=".",ms=8,fmt='.k',alpha=0.2,elinewidth=1)
plt.title(r'$\rm{Distance\ Modulus\ Residual}$',fontsize=15)#,y=1.02)
plt.xlabel(r'$\rm{Redshift\ (z)}$',fontsize=14)
plt.ylabel(r'$\rm{\Delta \mu (z)\ (mag)}$',fontsize=14)
plt.xlim([0,1.2])

plt.tight_layout()
plt.savefig('../figures/bias_subplots_fd1_nobias_er.png')
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
    
    
c = np.loadtxt('../data/bias_chain_fd1_nobias_er.txt',usecols=[0,2])
N100,N95,N68,X,Y,Z = contour(c,[0,1], labels=['1', '2'],line=False)
col = ('#a3c0f6','#0057f6')

sns.set(style="white")#darkgrid, whitegrid, dark, white, ticks

plt.figure(num=None, figsize=(10, 8), dpi=80, facecolor='w', edgecolor='k')
gs = gridspec.GridSpec(2, 2, width_ratios=[3,1], height_ratios=[1,3])
ax=plt.subplot(gs[1,0])
plt.contourf(X,Y,Z,levels=[N95,N68,N100],colors=col,alpha=0.8)
plt.title(r'$\rm{MCMC\ Contours\ No\ Bias}$',fontsize=15)#,y=1.02)
plt.ylabel(r'$w$',fontsize=14)
plt.xlabel('$\Omega_M$',fontsize=14)
plt.ylim([-2.5,0])

plt.subplot(gs[0,0],sharex=ax)
plt.hist(om_nb,bins=50,normed=True,color=col[0],alpha=0.8)
plt.xlim([0,0.7])
plt.yticks([])

plt.subplot(gs[1,1],sharey=ax)
plt.hist(w_nb,bins=50,normed=True,orientation='horizontal',color=col[0],alpha=0.8)
plt.xticks([])

plt.savefig('../figures/bias_contour_fd1_nobias_er.png')
plt.close()
