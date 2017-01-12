# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 12:54:42 2016

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

def PI(x):
    """
    x: input parameter (array)
    returns the product of elements in the array"""
    
    y = funct.reduce(op.mul,x)
    return y


###############Data manipulation###############

z_obs,mu_obs,sig_mu_obs = np.loadtxt('../data/fakedata1_er.txt',
                                     usecols=[0,2,3],unpack=True)


###############MCMC code###############

n = 50000 #Number of steps
status=True #whether or not a statusbar is shown


###############MCMC code (no bias)###############

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

print('Generating unbiased posterior')

fout = open('../data/bias_chain_fd1_nobias_er.txt','w')
fout.write('#o_m \t H0 \t w \n')
for i in range(0,n-1):
    #current position:
    mu_theory1 = mu_interp(z_obs,om_current,H0_current,w_current)
    chi2_current = ((mu_obs-mu_theory1)/sig_mu_obs)**2
    
    likelihood_current = PI((1/np.sqrt(2*np.pi*sig_mu_obs**2))*np.exp(-chi2_current/2))
    prior = 1
    evidence = 1
    posterior_current = likelihood_current*prior/evidence
    
    om_proposed = om_current + rand.gauss(0,omstep)
    while om_proposed >= 1 or om_proposed <= 0:         #keeps Omega_Lamda in (0,1) 
        om_proposed = om_current + rand.gauss(0,omstep) #for numerical reasons
    H0_proposed = H0_current + rand.gauss(0,H0step)
    w_proposed = w_current + rand.gauss(0,wstep)
    
    #proposed position:
    mu_theory2 = mu_interp(z_obs,om_proposed,H0_proposed,w_proposed)
    chi2_proposed = ((mu_obs-mu_theory2)/sig_mu_obs)**2
    
    likelihood_proposed = PI((1/np.sqrt(2*np.pi*sig_mu_obs**2))*np.exp(-chi2_proposed/2))
    prior = 1
    evidence = 1
    posterior_proposed = likelihood_proposed*prior/evidence
    
    #decision:
    r = posterior_proposed/posterior_current
    
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