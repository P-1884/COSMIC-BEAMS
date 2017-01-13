# -*- coding: utf-8 -*-
"""
Created on Fri Jan 13 10:57:37 2017

@author: ethan
"""

import numpy as np
import random as rand
import sys
import operator as op
import functools as funct
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

mu_w_vectorized = np.vectorize(mu_w)

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

def PI(x):
    """
    x: input parameter (array)
    returns the product of elements in the array"""
    
    y = funct.reduce(op.mul,x)
    return y


###############Data manipulation###############

zt,z2,mu_obs,sig_mu_obs = np.loadtxt('../data/fakedata1_er.txt',
                                     usecols=[0,1,2,3],unpack=True)

"""
this loop shifts the z coordinate by a random amount given by the variable
'shift' with a random chance given by 'snpop'. it generates a contaminated
supernovae sample, as well as the split contaminant and non-contaminant 
samples.
z_obs0: contaminated sample
z_obs1: non-contaminants only
z_obs2: contaminants only"""

z_obs0 = []
z_obs1 = []
z_obs2 = []
z_obs3 = []
mu_obs0 = []
mu_obs1 = []
mu_obs2 = []
sig_mu_obs0 = []
sig_mu_obs1 = []
sig_mu_obs2 = []

missID = np.empty(len(zt))
snpop = 0.09
rand.seed(7)
for i in range(len(zt)):
    MC = rand.random()
    if zt[i] > 0.1 and MC < snpop:
        missID[i] = 1
        z_obs0.append(z2[i])
        z_obs2.append(z2[i])
        z_obs3.append(zt[i])
        mu_obs0.append(mu_obs[i])
        mu_obs2.append(mu_obs[i])
        sig_mu_obs0.append(sig_mu_obs[i])
        sig_mu_obs2.append(sig_mu_obs[i])
    else: 
        missID[i] = 0
        z_obs0.append(zt[i])
        z_obs1.append(zt[i])
        z_obs3.append(z2[i])
        mu_obs0.append(mu_obs[i])
        mu_obs1.append(mu_obs[i])
        sig_mu_obs1.append(sig_mu_obs[i])
        sig_mu_obs0.append(sig_mu_obs[i])

sig_mu_obs0 = np.array(sig_mu_obs0)


###############MCMC code###############

n = 50000 #Number of steps
status=True #whether or not a statusbar is shown


###############MCMC code (bias)###############

omlist = [] #create empty list
H0list = []
wlist = []

om_current = 0.3 #starting values
H0_current = 70
w_current = -1

omlist.append(om_current) #append first value to list
H0list.append(H0_current)
wlist.append(w_current)

omstep = 0.07 #step sizes
H0step = 1.3
wstep = 0.07

P_gamma = 0.91

print('Generating biased posterior')

fout = open('../data/zbeams_chain_ta_fd1_case3_er.txt','w')
fout.write('#o_m \t H0 \t w \n')
for i in range(0,n-1):
    #current position:
    mu_theory11 = mu_interp(z_obs0,om_current,H0_current,w_current)
    chi2_current1 = ((mu_obs-mu_theory11)/sig_mu_obs0)**2
    mu_theory12 = mu_interp(z_obs3,om_current,H0_current,w_current)
    chi2_current2 = ((mu_obs-mu_theory12)/sig_mu_obs0)**2
    
    L1_current = P_gamma*((1/np.sqrt(2*np.pi*sig_mu_obs0**2))*np.exp(-chi2_current1/2))
    L2_current = (1-P_gamma)*((1/np.sqrt(2*np.pi*sig_mu_obs0**2))*np.exp(-chi2_current2/2))
    likelihood_current = L1_current + L2_current
    prior = 1
    evidence = 1
    posterior_current = likelihood_current*prior/evidence
    
    log_like_current = sum(np.log(likelihood_current))
    
    om_proposed = om_current + rand.gauss(0,omstep)
    while om_proposed >= 1 or om_proposed <= 0:         #keeps Omega_Lamda in (0,1) 
        om_proposed = om_current + rand.gauss(0,omstep) #for numerical reasons
    H0_proposed = H0_current + rand.gauss(0,H0step)
    while H0_proposed >= 200 or H0_proposed <= 10:
        H0_proposed = H0_current + rand.gauss(0,H0step)
    w_proposed = w_current + rand.gauss(0,wstep)
    while w_proposed >= 4 or w_proposed <= -6:
        w_proposed = w_current + rand.gauss(0,wstep)
    
    #proposed position:
    mu_theory21 = mu_interp(z_obs0,om_proposed,H0_proposed,w_proposed)
    chi2_proposed1 = ((mu_obs-mu_theory21)/sig_mu_obs0)**2
    mu_theory22 = mu_interp(z_obs3,om_proposed,H0_proposed,w_proposed)
    chi2_proposed2 = ((mu_obs-mu_theory22)/sig_mu_obs0)**2
    
    L1_proposed = P_gamma*((1/np.sqrt(2*np.pi*sig_mu_obs0**2))*np.exp(-chi2_proposed1/2))
    L2_proposed = (1-P_gamma)*((1/np.sqrt(2*np.pi*sig_mu_obs0**2))*np.exp(-chi2_proposed2/2))
    likelihood_proposed = L1_proposed + L2_proposed
    prior = 1
    evidence = 1
    posterior_proposed = likelihood_proposed*prior/evidence
    
    log_like_proposed = sum(np.log(likelihood_proposed))
    
    #decision:
    #r = posterior_proposed/posterior_current
    r = np.exp(log_like_proposed - log_like_current)
    
    MC = rand.random()
    
    if r < 1 and MC <= r:
        omlist.append(om_proposed)
        H0list.append(H0_proposed)
        wlist.append(w_proposed)
    elif r < 1 and MC > r:
        omlist.append(om_current)
        H0list.append(H0_current)
        wlist.append(w_current)
    else:
        omlist.append(om_proposed)
        H0list.append(H0_proposed)
        wlist.append(w_proposed)
        
    om_current = omlist[i+1]
    H0_current = H0list[i+1]
    w_current = wlist[i+1]
    
    fout.write(str(om_current)+'\t'+str(H0_current)+'\t'+str(w_current)+'\n')
    
    if status==True:
        inc100 = np.int64(i/(n-2)*100)
        inc50 = np.int64(i/(n-2)*50)
        sys.stdout.write('\r')
        sys.stdout.write('[' + '#'*inc50 + ' '*(50-inc50) + ']' + str(inc100) + '%')
        sys.stdout.flush()

fout.close()
print('\ndone')