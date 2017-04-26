#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 22:10:51 2017

@author: ethan
"""

import numpy as np

dat0 = np.loadtxt('fakedata_photometric.txt',usecols=[0],unpack=True)
dat1 = np.loadtxt('mcmc_chain_bi_b1.txt',unpack=False)

dat1 = np.delete(dat1,[0,1,2,3],1)

fout0 = open('del_z_chains.txt','w') #outputs z_i-z_true for each chain

for i in range(len(dat1)):
    for j in range(len(dat0)):
        fout0.write(str(dat1[i][j]-dat0[j])+'\t')
    fout0.write('\n')
fout0.close()


fout1 = open('zchain_values.txt','w') #outputs mean and std for each chain
for i in range(4,len(dat1.T)-2):
    fout1.write(str(np.mean(dat1.T[i]))+'\t'+str(np.std(dat1.T[i]))+'\n')
fout1.close()
