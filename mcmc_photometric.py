# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 22:29:47 2017

@author: ethan
"""

import numpy as np
import random as rand
import sys

from zbeamsfunctions import likelihood_phot


###############MCMC code###############

n = 500000 #Number of steps

bias = sys.argv[1]
if bias == 'bi':
    column = 1
elif bias =='nb':
    column = 0
block = sys.argv[2]

dz = 0.04
status = True
burnin = 45000
thinning = 10

z,mu,sig_mu = np.loadtxt('fakedata_photometric.txt',usecols=[column,2,3],unpack=True)
fout = open('mcmc_chain_'+bias+'_b'+block+'.txt','w')
fout.write('#OM \t\t\t H0 \t\t\t w \n')

zdisp = dz*(1+z.copy())
block = int(block)

OMlist = [] #create empty list
H0list = []
wlist = []
accept_list = []

OM_current = 0.31 #starting values
H0_current = 67.74
w_current = -1
z_current = z.copy()
b_current = 3

OMlist.append(OM_current) #append first value to list
H0list.append(H0_current)
wlist.append(w_current)

bstep = 0

print('Generating posterior')
accept = 0
log_like_proposed = 0
lenz = len(z)
rand.seed(7)
for i in range(0,n-1):
    #current position:
    MC2 = rand.random()
    if i <= burnin:
        OMstep = 0 #step sizes
        H0step = 0
        wstep = 0
        zstep = 1*1.0
    else:
        OMstep = 1.0*0.059 #step sizes
        H0step = 1.0*0.793
        wstep = 1.0*0.175
        zstep = 2*0.9
        bstep = 0.0
    
    if i == 0:
        log_like_current = likelihood_phot(z_current,mu,sig_mu,OM_current,H0_current,w_current,z,zdisp,b_current)
    elif accept == 1:
        log_like_current = log_like_proposed
    
    OM_proposed =  rand.gauss(OM_current,OMstep)
    while OM_proposed >= 1 or OM_proposed <= 0:         #keeps Omega_matter in (0,1) 
        OM_proposed =  rand.gauss(OM_current,OMstep)    #for numerical reasons
    H0_proposed =  rand.gauss(H0_current,H0step)
    while H0_proposed >= 200 or H0_proposed <= 10:
        H0_proposed =  rand.gauss(H0_current,H0step)
    w_proposed =  rand.gauss(w_current,wstep)
    while w_proposed >= 4 or w_proposed <= -6:
        w_proposed =  rand.gauss(w_current,wstep)
    
    MC1 = [rand.randint(0,lenz-1) for j in range(block)]
    z_proposed = z_current.copy()
    z_proposed[MC1] = [rand.gauss(z_current[MC1],(zstep*zdisp[MC1])**2) for j in range(block)]
    
    b_proposed = rand.gauss(b_current,bstep)
    
    #proposed position:
    log_like_proposed=likelihood_phot(z_proposed,mu,sig_mu,OM_proposed,H0_proposed,w_proposed,z,zdisp,b_proposed)
    
    #decision:
    r = np.exp(log_like_proposed - log_like_current)
    MC0 = rand.random()
    
    if r < 1 and MC0 > r:
        OMlist.append(OM_current)
        H0list.append(H0_current)
        wlist.append(w_current)
        accept = 0
        like = log_like_current
    else:
        OMlist.append(OM_proposed)
        H0list.append(H0_proposed)
        wlist.append(w_proposed)
        z_current = z_proposed.copy()
        b_current = b_proposed
        accept = 1
        like = log_like_proposed
    
    accept_list.append(accept)
    OM_current = OMlist[i+1]
    H0_current = H0list[i+1]
    w_current = wlist[i+1]
    
    if i >= burnin and i%thinning == 0:
        fout.write(str(OM_current)+'\t'+str(H0_current)+'\t'+str(w_current)+'\t'+str(like))
        if block != 0 and bias == 'bi':
            for j in range(len(z_current)):
                fout.write('\t'+str(z_current[j]))
        fout.write('\n')
    
    if status==True:
        inc100 = np.int64(i/(n-2)*100)
        inc50 = np.int64(i/(n-2)*50)
        sys.stdout.write('\r')
        sys.stdout.write('[' + '#'*inc50 + ' '*(50-inc50) + ']' + str(inc100) + '%')
        sys.stdout.flush()
        
fout.close()
print('\ndone')