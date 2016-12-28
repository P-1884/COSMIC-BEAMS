# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 12:33:14 2016

@author: ethan
"""

import random as rand
import numpy as np
from scipy.integrate import quad

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


###############Cosmology###############

H0 = 67.74
OM = 0.31
w = -1


###############Sample parameters###############

rand.seed(69621)
"""
fakedata1.txt has rand.seed(16807)
fakedata2.txt has rand.seed(48271)
fakedata3.txt has rand.seed(69621)"""
n = 1000 #sample size

z1 = 0.015
zn = 1.415
mu_disp = 0.2
sig_mu_value = 0.2


###############Generate sample###############

z = np.empty(n)
mu = np.empty(n)
sig_mu = np.empty(n)

for i in range(n):
    z[i] = z1 + (zn-z1)*rand.random()
    mu[i] = mu_w(z[i],OM,H0,w) + rand.gauss(0,mu_disp)
    sig_mu[i] = sig_mu_value


###############Write output to file###############

fout = open('../data/fakedata3.txt','w')

for i in range(n):
    fout.write(str(z[i])+'\t'+str(mu[i])+'\t'+str(sig_mu[i])+'\n')

fout.close