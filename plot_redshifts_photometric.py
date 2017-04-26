# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 22:30:31 2017

@author: ethan
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


dat = np.loadtxt('del_z_chains.txt',unpack=False)
zt = np.loadtxt('fakedata_photometric.txt',usecols=[0],unpack=True)

sns.set(style="white")#darkgrid, whitegrid, dark, white, ticks

plt.figure(figsize=(10,5), dpi=80, facecolor='w')
for i in range(1000):
    nbins = round((np.abs(np.max(dat.T[i]))+np.abs(np.min(dat.T[i])))*70)
    plt.hist(dat.T[i],bins=nbins,normed=True,alpha=0.1,color=[zt[i]/zt.max(),0,0])
plt.ylim([0,140])
plt.xlabel(r'$\rm{z-z_{true}}$',fontsize=19)
plt.ylabel(r'$\rm{P(z-z_{true})}$',fontsize=19)
plt.tick_params(axis='both', which='major', labelsize=13)
plt.xticks(fontname='Serif')
plt.yticks(fontname='Serif')

plt.tight_layout()
plt.savefig('zposteriors_hist.png')
plt.close()