# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 11:02:50 2017

@author: ethan
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from zbeamsfunctions import mu_w


"""
for this code to run, the parent folder should look like:
parent_dir/
    this_script.py
    zbeamsfunctions.py
    fakedata_photometric.txt
    zchain_values.txt

to do this, both mcmc_photometric.py and del_z_chains.py should be run first.
"""

z2,muo,sig_mu = np.loadtxt('fakedata_photometric.txt',usecols=[1,2,3],unpack=True)

zt,sig_zt = np.loadtxt('zchain_values.txt',unpack=True)

sig_z2 = 0.04*(1+z2)

z = np.linspace(0.015,1.4,1000)
mu_w_vec = np.vectorize(mu_w)
mu = mu_w_vec(z,0.31,67.74,-1)

sns.set(style="white")

# the main axes is subplot(111) by default
plt.figure(figsize=(10,6.5), dpi=80, facecolor='w')
(_, caps, _) = plt.errorbar(z2,muo-mu_w_vec(z2,0.31,67.74,-1),yerr=sig_mu,xerr=sig_z2,fmt='.',
                            marker='d',ms=6,ls='',color='chocolate',alpha=0.74,ecolor='sandybrown',
                            capsize=4,errorevery=1,zorder=1,elinewidth=1)
for cap in caps:
    cap.set_markeredgewidth(1)
plt.plot([0,1.6],[0,0],color='k')
plt.axis([0, 1.6, -3, 6])
plt.xlabel(r'$\rm{z_{obs}}$',fontsize=19)
plt.ylabel(r'$\rm{\Delta \mu}$',fontsize=19)
plt.tick_params(axis='both', which='major', labelsize=13)
plt.xticks(fontname='Serif')
plt.yticks(fontname='Serif')

# this is an inset axes over the main axes
a = plt.axes([0.54, 0.54, 0.32, 0.32])
(_, caps, _) = plt.errorbar(zt,muo-mu_w_vec(zt,0.31,67.74,-1),yerr=sig_mu,xerr=sig_zt,fmt='.',
                            marker='d',ms=3,ls='',color='b',alpha=0.34,ecolor='steelblue',capsize=2,
                            errorevery=1,zorder=1,elinewidth=1)
for cap in caps:
    cap.set_markeredgewidth(1)
plt.plot([0,1.6],[0,0],color=[0,0,0])
plt.xlim(0, 1.5)
plt.xlabel(r'$\rm{\bar{z}}$',fontsize=19)
plt.ylabel(r'$\rm{\Delta \mu}$',fontsize=19)
plt.tick_params(axis='both', which='major', labelsize=13)
plt.xticks([0,0.5,1,1.5],fontname='Serif')
plt.yticks(fontname='Serif')

plt.savefig('hubble_residual_photometric.png')
plt.close()