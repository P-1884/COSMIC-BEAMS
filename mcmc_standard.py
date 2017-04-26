#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 22:39:54 2017

@author: ethan
"""

import numpy as np
import random as rand
import sys
from scipy.integrate import quad
from scipy import interpolate

"""
this code runs assuming the following directory setup:
parent_dir/
    fakedata_spectroscopic.txt
    this_script.py
"""

###############Inputs###############

ds = sys.argv[1]
bias = sys.argv[2]
if bias == 'nb':
    column = 0
    biasname='nobias'
elif bias == 'bi':
    column = 1
    biasname='bias'


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


###############Data manipulation###############

zt,mu_obs0 = np.loadtxt('fakedata_spectroscopic.txt',usecols=[column,4],unpack=True)

rand.seed(7)

z_obs0 = zt


###############MCMC code###############

n = 100000 #Number of steps
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

omstep = 0.5*0.059 #step sizes
H0step = 0.5*0.793
wstep = 0.5*0.157

P_gamma = 0.91
P_tau = 0.95
offset = 2
sig_tau1 = 0.2
sig_tau2 = 1.5

accept = 0
log_like_proposed = 0
accept_list = []

print('Generating biased posterior')

fout = open('../data/mcmc_chain_'+biasname+'.txt','w')
fout.write('#o_m \t H0 \t w \n')
for i in range(0,n-1):
    #current position:
    if i == 0:
        mu_theory11 = mu_interp(z_obs0,om_current,H0_current,w_current)
        chi2_current1 = ((mu_obs0-mu_theory11)/sig_tau1)**2
        
        likelihood_current = (1/np.sqrt(2*np.pi*sig_tau1**2))*np.exp(-chi2_current1/2)
        
        log_like_current = sum(np.log(likelihood_current))
    elif accept == 1:
        log_like_current = log_like_proposed
    
    om_proposed =  rand.gauss(om_current,omstep)
    while om_proposed >= 1 or om_proposed <= 0:         #keeps Omega_matter in (0,1) 
        om_proposed =  rand.gauss(om_current,omstep)    #for numerical reasons
    H0_proposed =  rand.gauss(H0_current,H0step)
    while H0_proposed >= 200 or H0_proposed <= 10:
        H0_proposed =  rand.gauss(H0_current,H0step)
    w_proposed =  rand.gauss(w_current,wstep)
    while w_proposed >= 4 or w_proposed <= -6:
        w_proposed =  rand.gauss(w_current,wstep)
    
    #proposed position:
    mu_theory21 = mu_interp(z_obs0,om_proposed,H0_proposed,w_proposed)
    chi2_proposed1 = ((mu_obs0-mu_theory21)/sig_tau1)**2
    
    likelihood_proposed = (1/np.sqrt(2*np.pi*sig_tau1**2))*np.exp(-chi2_proposed1/2)
    
    log_like_proposed = sum(np.log(likelihood_proposed))
    
    #decision:
    r = np.exp(log_like_proposed - log_like_current)
    
    MC = rand.random()
    
    if r < 1 and MC > r:
        omlist.append(om_current)
        H0list.append(H0_current)
        wlist.append(w_current)
        accept = 0
    else:
        omlist.append(om_proposed)
        H0list.append(H0_proposed)
        wlist.append(w_proposed)
        accept = 1
        
    om_current = omlist[i+1]
    H0_current = H0list[i+1]
    w_current = wlist[i+1]
    accept_list.append(accept)
    
    fout.write(str(om_current)+'\t'+str(H0_current)+'\t'+str(w_current)+'\n')
    
    if status==True:
        inc100 = np.int64(i/(n-2)*100)
        inc50 = np.int64(i/(n-2)*50)
        sys.stdout.write('\r')
        sys.stdout.write('[' + '#'*inc50 + ' '*(50-inc50) + ']' + str(inc100) + '%')
        sys.stdout.flush()

fout.close()
print('\ndone')