# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 00:37:13 2017

@author: ethan
"""

import numpy as np
import random as rand
import sys
import operator as op
import functools as funct

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

def yfunc(x,m,c):
    return m*x+c

def dyfunc(x,m,c):
    return m

def PI(x):
    """
    x: input parameter (array)
    returns the product of elements in the array"""
    
    y = funct.reduce(op.mul,x)
    return y


###############Data manipulation###############

taylor = True
bias = True
status = False #whether or not a statusbar is shown
x,y,sig_y = np.loadtxt('../data/fakedata_linear_er.txt',usecols=[0,1,2],unpack=True)

x0 = []
x1 = []
x2 = []
y0 = []
y1 = []
y2 = []
sig_y0 = []
sig_y1 = []
sig_y2 = []

if bias == True:
    shift = 0.1
else: shift = 0

"""
this loop shifts the z coordinate by a random amount given by the variable
'shift' with a random chance given by 'snpop'. it generates a contaminated
supernovae sample, as well as the split contaminant and non-contaminant 
samples.
z_obs0: contaminated sample
z_obs1: non-contaminants only
z_obs2: contaminants only"""

rand.seed(7)
for i in range(0,len(x)):
    MC = rand.random()
    if x[i] < 0.1:
        x0.append(x[i])
        x1.append(x[i])
        y0.append(y[i])
        y1.append(y[i])
        sig_y1.append(sig_y[i])
        sig_y0.append(sig_y[i])
    else:
        shifted_x = x[i]-(rand.gauss(0,shift*(1+x[i]))) #SHOULD I USE 1+x[i] OR JUST x[i]????
        x0.append(shifted_x)
        x2.append(shifted_x)
        y0.append(y[i])
        y2.append(y[i])
        sig_y0.append(sig_y[i])
        sig_y2.append(sig_y[i])
            
sig_y0 = np.array(sig_y0)
x0 = np.array(x0)


###############MCMC code###############

n = 50000 #Number of steps


###############MCMC code (bias)###############

mlist = [] #create empty list
clist = []
xbarlist = []
Rxlist = []

m_current = 1 #starting values
c_current = 1
xbar_current = 5
Rx_current = 3

mlist.append(m_current) #append first value to list
clist.append(c_current)
xbarlist.append(xbar_current)
Rxlist.append(Rx_current)

mstep = 0.01 #step sizes
cstep = 0.01
xbarstep = 2.0
Rxstep = 0.5

print('Generating biased posterior')

fout = open('../data/bias_chain_linear_taylor_gausszprior_er.txt','w')
#a refers to the errorbar compensation (i.e. none)
#b refers to the errorbar compensation (i.e. outside the loop with fiducial model)
#c refers to the errorbar compensation (i.e. inside the loop with fiducial model)
fout.write('#m \t c \n')
for i in range(0,n-1):
    #current position:
    y_theory1 = yfunc(x0,m_current,c_current)
    if taylor == True:
        dy = dyfunc(x0,m_current,c_current)
        new_err = np.sqrt(sig_y0**2 + (dy*shift*(1+x0))**2)
        chi2_current = ((y-y_theory1)/new_err)**2
        P1 = sum(np.log(1/(np.sqrt(2*np.pi*np.ones(len(x0))*Rx_current))))
        P2 = 0.5*sum(((x0-xbar_current)/Rx_current)**2)
        prior_current = P1 - P2
        log_like_current = sum(np.log(1/(np.sqrt(2*np.pi*new_err**2))))-0.5*sum(chi2_current)+prior_current
    else: 
        chi2_current = ((y-y_theory1)/sig_y0)**2
        log_like_current = sum(np.log(1/(np.sqrt(2*np.pi*sig_y0**2))))-0.5*sum(chi2_current)
    """
    likelihood_current = PI((1/np.sqrt(2*np.pi*sig_mu_obs0**2))*np.exp(-chi2_current/2))
    prior = 1
    evidence = 1
    posterior_current = likelihood_current*prior/evidence
    """
    
    m_proposed = m_current + rand.gauss(0,mstep)
    c_proposed = c_current + rand.gauss(0,cstep)
    xbar_proposed = xbar_current + rand.gauss(0,xbarstep)
    Rx_proposed = Rx_current + rand.gauss(0,Rxstep)
    
    #proposed position:
    y_theory2 = yfunc(x0,m_proposed,c_proposed)
    if taylor == True:
        dy = dyfunc(x0,m_proposed,c_proposed)
        new_err = np.sqrt(sig_y0**2 + (dy*shift*(1+x0))**2)
        chi2_proposed = ((y-y_theory2)/new_err)**2
        P3 = sum(np.log(1/(np.sqrt(2*np.pi*np.ones(len(x0))*Rx_proposed))))
        P4 = 0.5*sum(((x0-xbar_proposed)/Rx_proposed)**2)
        prior_proposed = P3-P4
        log_like_proposed = sum(np.log(1/(np.sqrt(2*np.pi*new_err**2))))-0.5*sum(chi2_proposed)+prior_proposed
    else: 
        chi2_proposed = ((y-y_theory2)/sig_y0)**2
        log_like_proposed = sum(np.log(1/(np.sqrt(2*np.pi*sig_y0**2))))-0.5*sum(chi2_proposed)
    """
    likelihood_proposed = PI((1/np.sqrt(2*np.pi*sig_mu_obs0**2))*np.exp(-chi2_proposed/2))
    prior = 1
    evidence = 1
    posterior_proposed = likelihood_proposed*prior/evidence
    """
    
    #decision:
    #r = posterior_proposed/posterior_current
    r = np.exp(log_like_proposed - log_like_current)
    
    MC = rand.random()
    
    if r < 1 and MC > r:
        mlist.append(m_current)
        clist.append(c_current)
        xbarlist.append(xbar_current)
        Rxlist.append(Rx_current)
    else:
        mlist.append(m_proposed)
        clist.append(c_proposed)
        xbarlist.append(xbar_proposed)
        Rxlist.append(Rx_proposed)
        
    m_current = mlist[i+1]
    c_current = clist[i+1]
    xbar_current = xbarlist[i+1]
    Rx_current = Rxlist[i+1]
    
    fout.write(str(m_current)+'\t'+str(c_current)+'\n')
    
    if status==True:
        inc100 = np.int64(i/(n-2)*100)
        inc50 = np.int64(i/(n-2)*50)
        sys.stdout.write('\r')
        sys.stdout.write('[' + '#'*inc50 + ' '*(50-inc50) + ']' + str(inc100) + '%')
        sys.stdout.flush()

fout.close()
print('\ndone')