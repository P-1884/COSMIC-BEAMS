# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 22:30:31 2017

@author: ethan
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib import gridspec
import seaborn as sns

from zbeamsfunctions import contour


###############MCMC points###############

nb_mcmc_file = 'mcmc_chain_nobias.txt'
bi_mcmc_file = 'mcmc_chain_bias.txt'
zb_mcmc_file = 'mcmc_chain_zbeams.txt'

om_nb,H0_nb,w_nb = np.loadtxt(nb_mcmc_file, usecols=[0,1,2],unpack=True)
om_bi,H0_bi,w_bi = np.loadtxt(bi_mcmc_file, usecols=[0,1,2],unpack=True)
om_zb,H0_zb,w_zb = np.loadtxt(zb_mcmc_file, usecols=[0,1,2],unpack=True)


###############Contour plot###############

c_nb = np.loadtxt(nb_mcmc_file,usecols=[0,2])
c_bi = np.loadtxt(bi_mcmc_file,usecols=[0,2])
c_zb = np.loadtxt(zb_mcmc_file,usecols=[0,2])
N100_nb,N95_nb,N68_nb,X_nb,Y_nb,Z_nb = contour(c_nb,[0,1],labels=['1','2'],smooth=21e5,line=True)
N100_bi,N95_bi,N68_bi,X_bi,Y_bi,Z_bi = contour(c_bi,[0,1],labels=['1','2'],smooth=20e5,line=False)
N100_zb,N95_zb,N68_zb,X_zb,Y_zb,Z_zb = contour(c_zb,[0,1],labels=['1','2'],smooth=16e5,line=False)
col0 = ('#a3c0f6','#0057f6')#light blue;dark blue
col1 = ('#fc9272','#de2d26')#light red;dark red
col3 = ('k','k')

sns.set(style="white")#darkgrid, whitegrid, dark, white, ticks

plt.figure(num=None, figsize=(10, 8), dpi=80, facecolor='w', edgecolor='k')
gs = gridspec.GridSpec(2, 2, width_ratios=[3,1], height_ratios=[1,3])
ax=plt.subplot(gs[1,0])
cs0 = plt.contourf(X_bi,Y_bi,Z_bi,levels=[N95_bi,N68_bi,N100_bi],colors=col1,alpha=0.8,hatches=['//'])
cs1 = plt.contourf(X_zb,Y_zb,Z_zb,levels=[N95_zb,N68_zb,N100_zb],colors=col0,alpha=0.8)
cs2 = plt.contour(X_nb,Y_nb,Z_nb,levels=[N95_nb,N68_nb,N100_nb],colors=col3,alpha=1.0,linewidths=1)
cs3 = plt.scatter(0.31,-1,s=50,linewidth=1,marker='x',c='k',label=r'$\rm{Fiducial\ Model}$')
plt.ylabel(r'${w}$',fontsize=17+2)
plt.xlabel(r'$\rm{\Omega_m}$',fontsize=17+2)
plt.tick_params(axis='both', which='major', labelsize=13)
plt.xticks(fontname='Serif')
plt.yticks(fontname='Serif')

patch7 = mpatches.Patch(color=col0[0], label=r'$\rm{zBEAMS}$')
patch8 = mpatches.Patch(color=col1[0], label=r'$\rm{Biased}$',hatch='/',ls='solid',lw=1)
patch9 = mlines.Line2D([],[],color='k',linewidth=1, label=r'$\rm{Spectroscopic}$')
plt.legend(handles=[cs3,patch9,patch8,patch7],loc=3,fontsize=13+2)

plt.subplot(gs[0,0],sharex=ax)
plt.hist(om_bi,bins=50,normed=True,color=col1[0],alpha=0.6)
plt.hist(om_zb,bins=50,normed=True,color=col0[0],alpha=0.6)
plt.xlim([0,0.5])
plt.tick_params(axis='both', which='major', labelsize=13)
plt.xticks(fontname='Serif')
plt.yticks([])

plt.subplot(gs[1,1],sharey=ax)
plt.hist(w_bi,bins=30,normed=True,orientation='horizontal',color=col1[0],alpha=0.6)
plt.hist(w_zb,bins=45,normed=True,orientation='horizontal',color=col0[0],alpha=0.6)
plt.ylim([-1.8,-0.2])
plt.tick_params(axis='both', which='major', labelsize=13)
plt.yticks(fontname='Serif')
plt.xticks([])

plt.savefig('contours_spectroscopic.png')
plt.close()