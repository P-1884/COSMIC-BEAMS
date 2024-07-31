from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as pl

label_dict = {'OM':'$\Omega_m$','Ode':'$\Omega_\lambda$','Ok':'$\Omega_k$',
            'w':'$w_0$','wa':'$w_a$','H0':'H0',
            'mu_zL_g_L':'$\mu_{zL|L}$','mu_zL_g_NL':'$\mu_{zL|NL}$',
            'mu_zS_g_L':'$\mu_{zS|L}$','mu_zS_g_NL':'$\mu_{zS|NL}$',
            'sigma_zL_g_L':'$\sigma_{zL|L}$','sigma_zL_g_NL':'$\sigma_{zL|NL}$',
            'sigma_zS_g_L':'$\sigma_{zS|L}$','sigma_zS_g_NL':'$\sigma_{zS|NL}$',
            't':'Test Sampler','t_mu':'Test Sampler ($\mu$)'}

def fit_straight_line_to_samples(samples):
    def y(x,m,c):
        return m*x+c
    popt,_ = curve_fit(y,np.arange(len(samples)),samples)
    return y(np.arange(len(samples)),*popt),popt[0],popt[1]

def plot_JAX_chains(JAX_chains,fig=None,ax=None,zero=0,title=None,plot_hist=False,
                    ignore_nonconverged=False,exclude_list = [],fit_straight_line=True,color_0=None):
    if fig is None: fig,ax = pl.subplots(1+plot_hist,5,figsize=(25,5*(1+plot_hist)))
    column_list = JAX_chains.columns
    color_list = pl.rcParams['axes.prop_cycle'].by_key()['color'] #Default colour list.
    #rfind finds the index of the last occurrence of the character in a string: 
    column_set = list(set([elem[:elem.rfind('_')] for elem in column_list]))
    column_dict = {"OM":0,"Ode":1,'w':2,'wa':3,'Ok':4}#,'mu_zL_g_L':4}
    range_dict = {'OM':(-0.1,1.1),'Ode':(-0.1,1.1),'wa':(-3,1),'w':(-3,4),'Ok':(-1.1,1.1)}
    bin_dict = {'OM':(0,1),'Ode':(0,1),'wa':(-3,1),'w':(-3,4),'Ok':(-1,1)}
    for p_i,c_i in enumerate(JAX_chains.keys()):
        c_i_set = c_i[:c_i.rfind('_')]
        chain_number = int(c_i.replace(f'{c_i_set}_',''))
        if chain_number in exclude_list: print(f'Excluding {c_i}');continue
        if color_0 is None: color = color_list[chain_number%len(color_list)]
        else: color = color_0
        try: column_dict[c_i_set]
        except:continue
        try:
            if plot_hist: 
                ax_0 = ax[0,column_dict[c_i_set]]
            else: ax_0 = ax[column_dict[c_i_set]]
            ax_0.plot(zero+np.arange(len(JAX_chains[c_i])),np.array(JAX_chains[c_i]),
                                         alpha=0.5,label=c_i,color=color)
            if fit_straight_line:
                Best_Fit, m, c = fit_straight_line_to_samples(np.array(JAX_chains[c_i]))
                ax_0.plot(Best_Fit,'k--')
                Y_0 = np.round(c,2);Y_1 = np.round(m*len(JAX_chains[c_i])+c,2)
                ax_0.text(0,1.1*Y_0,Y_0)
                ax_0.text(len(JAX_chains[c_i]),1.1*Y_1,Y_1)
            ax_0.set_title(label_dict[c_i_set],fontsize=18,fontweight='bold')
            ax_0.set_ylim(range_dict[c_i_set])
            ax_0.legend(loc='lower right')
        except Exception as ex:
            pass 
    #Have to do second loop to plot the histograms with equal bins:
    for p_i,c_i in enumerate(JAX_chains.keys()):
        #print(c_i,c_i_set)
        c_i_set = c_i[:c_i.rfind('_')]
        chain_number = int(c_i.replace(f'{c_i_set}_',''))
        if chain_number in exclude_list: print(f'Excluding {c_i}');continue
        if color_0 is None: color_i = color_list[chain_number%len(color_list)]
        else: color_i = color_0
        try: column_dict[c_i_set]
        except: continue
        if plot_hist: 
            ax_1 = ax[1,column_dict[c_i_set]]
            #limits_i = ax[0,column_dict[c_i_set]].get_ylim()
            bins_i = np.linspace(bin_dict[c_i_set][0],bin_dict[c_i_set][1],20)
            color_i = color_list[chain_number%len(color_list)]
            if ignore_nonconverged:
                if len(set(np.array(JAX_chains[c_i])))<10: continue
            ax_1.hist(np.array(JAX_chains[c_i]),fill=False,edgecolor=color_i,bins = bins_i,linewidth=3)
            ax_1.set_xlabel(label_dict[c_i_set],fontsize=15)
            ax_1.set_xlim(range_dict[c_i_set])
            #ax[column_dict[c_i_set]].set_title(c_i)        
    #pl.suptitle(f'Key: {k_i}')
    if title is not None: pl.suptitle(title)
    if fig is None: pl.show()
    #pl.close()

pl.close()