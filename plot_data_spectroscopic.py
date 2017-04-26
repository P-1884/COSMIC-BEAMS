# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 22:30:31 2017

@author: ethan
"""

import numpy as np
import matplotlib.pyplot as plt

from zbeamsfunctions import mu_w

"""
for this code to run, the parent folder should look like:
parent_dir/
    this_script.py
    zbeamsfunctions.py
    fakedata_spectroscopic.txt
"""


z_o,mu_o,c_type,c_z = np.loadtxt('fakedata_spectroscopic.txt',usecols=[1,4,5,6],unpack=True)

mu_w_vec = np.vectorize(mu_w)
z = np.linspace(0.01,1,1000)
mu_t = mu_w_vec(z,0.31,67.74,-1)

plt.figure(figsize=(10,6.5), dpi=80, facecolor='w', edgecolor='k')

for i in range(len(z_o)):
    if c_type[i] == 1:
        if c_z[i] == 1:
            plt.errorbar(z_o[i],mu_o[i],yerr=0.2,marker='',label=r'$\rm{Ia,\ correct\ host}$',
                         ls='',color='darkorange', alpha = 0.55, markersize=7, capsize = 3)
            if i == 0:
                plt.errorbar(2,42,yerr=0.2,marker='^',label=r'$\rm{Ia,\ wrong host}$',
                             ls='',color='darkred',markeredgecolor = 'none', markersize=6, capsize = 3)
                plt.errorbar(2,42,yerr=0.2,marker='s',label=r'$\rm{non-Ia,\ wrong\ host}$',
                             ls='',color='r',markeredgecolor = 'none', markersize=8, capsize = 3)
                plt.errorbar(2,42,yerr=0.2,marker='v',label=r'$\rm{non-Ia,\ correct\ host}$',
                             ls='',color='steelblue',markeredgecolor = 'none', markersize=7, capsize = 3)
        else:
            plt.errorbar(z_o[i],mu_o[i],yerr=0.2,marker='^',label=r'$\rm{Ia,\ wrong host}$',
                         ls='',color='darkred',markeredgecolor = 'none', markersize=6, capsize = 3)
    else:
        if c_z[i] == 0:
            plt.errorbar(z_o[i],mu_o[i],yerr=0.2,marker='s',label=r'$\rm{non-Ia,\ wrong\ host}$',
                         ls='',color='r',markeredgecolor = 'none', markersize=8, capsize = 3)
        else:
            plt.errorbar(z_o[i],mu_o[i],yerr=0.2,marker='v',label=r'$\rm{non-Ia,\ correct\ host}$',
                         ls='',color='steelblue',markeredgecolor = 'none', markersize=7, capsize = 3)
    if i == 0:
        plt.plot(z,mu_t,'--k',label=r'$\rm{Fid\ Ia}$',lw=3)
        plt.plot(z,mu_t+2,'-.',color='steelblue',label=r'$\rm{Fid\ non-Ia}$',lw=3, alpha = 0.5)
        plt.legend(loc=4,fontsize=15,frameon=False)

plt.xlabel(r'$\rm{z_{obs}}$', fontsize = 19)
plt.ylabel(r'$\rm{\mu}$', fontsize = 19)
plt.tick_params(axis='both', which='major', labelsize=13)
plt.xticks(fontname='Serif')
plt.yticks(fontname='Serif')
plt.xlim([0.01,1])

plt.tight_layout()
plt.savefig('hubble_diagram_spectroscopic.png')
plt.close()