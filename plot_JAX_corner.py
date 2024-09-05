import numpy as np
import matplotlib.pyplot as pl
import corner

range_dict = {'OM':(0,1),#(0.1,0.3),
              'Ode':(0,1),#(0.6,1),#(0.7,0.9),
              'Ok':(-1,1),
            'w':(-3.5,1.5),#(-1.4,-0.6),
            'wa':(-3.5,1.5),#(-3,1.5),
            'mu_zL_g_L':(0,1.5),'mu_zS_g_L':(0,3),
            'mu_zL_g_NL':(0,5),'mu_zS_g_NL':(0,5),
            'sigma_zL_g_L':(0,2),'sigma_zS_g_L':(0,2),
            'sigma_zL_g_NL':(0.1,5),'sigma_zS_g_NL':(0.1,5),
            'zL':(0,1.5),'zS':(0,3),'t':(0,2),'t_mu':(0,2),
            }
label_dict = {'OM':'$\Omega_m$','Ode':'$\Omega_\lambda$','Ok':'$\Omega_k$',
            'w':'$w_0$','wa':'$w_a$','H0':'H0',
            'mu_zL_g_L':'$\mu_{zL|L}$','mu_zL_g_NL':'$\mu_{zL|NL}$',
            'mu_zS_g_L':'$\mu_{zS|L}$','mu_zS_g_NL':'$\mu_{zS|NL}$',
            'sigma_zL_g_L':'$\sigma_{zL|L}$','sigma_zL_g_NL':'$\sigma_{zL|NL}$',
            'sigma_zS_g_L':'$\sigma_{zS|L}$','sigma_zS_g_NL':'$\sigma_{zS|NL}$',
            't':'Test Sampler','t_mu':'Test Sampler ($\mu$)',
            'alpha_mu_0':'$\\alpha_{\mu,0}$','alpha_mu_1':'$\\alpha_{\mu,1}$','alpha_mu_2':'$\\alpha_{\mu,2}$',
            'alpha_scale_0':'$\\alpha_{\sigma,0}$','alpha_scale_1':'$\\alpha_{\sigma,1}$','alpha_scale_2':'$\\alpha_{\sigma,2}$',
            'alpha_weights_0':'$\\alpha_{w,0}$','alpha_weights_1':'$\\alpha_{w,1}$','alpha_weights_2':'$\\alpha_{w,2}$',
            'scale_m':'$m_{scale}$','scale_c':'$c_{scale}$',
            's_m':'$m_{s}$','s_c':'$c_{s}$',
            }


def percentile_str(v,dp=2):
    perc_50 = np.round(np.percentile(v,50),dp).astype('str')
    perc_84_50 =  np.round(np.percentile(v,84)-np.percentile(v,50),dp).astype('str')
    perc_50_16 =  np.round(np.percentile(v,50)-np.percentile(v,16),dp).astype('str')
    return f'${perc_50}'+"_{-"+perc_50_16+"}^{+"+perc_84_50+'}$'

def plot_mu_sig(ax,v,c,y_frac=0.5):
    v=v.flatten()
    perc_50 = np.percentile(v,50)
    perc_84_50 =  float(np.percentile(v,84)-np.percentile(v,50))
    perc_50_16 =  float(np.percentile(v,50)-np.percentile(v,16))
    ylim = ax.get_ylim()
    ax.errorbar(perc_50,y_frac*ylim[1],xerr=np.array([[perc_50_16,perc_84_50]]).T,fmt='.',c=c)
    ax.set_ylim(ylim[0],1.2*ylim[1])

def plot_JAX_corner(sampler_list,
                    truth_dict={},range_dict={},label_dict = {},bin_dict = {},
                    key_list=[],legend_list=[],plot_Ok=False,fig=None,ax=[],burnin=np.nan,hist_ylim=[],
                    exclude_walker_list=[],add_text=True,color_list = ['darkred','darkgreen','purple','darkblue','darkorange','magenta',
                                                                       'red','green','blue','orange','black','cyan','yellow','brown','pink','grey'],
                    alpha_hist2d=1.0,alpha_hist1d=1.0):
    if not isinstance(sampler_list, list):
        print('Making into a list')
        sampler_list=[sampler_list]
    if len(key_list)==0: #Plotting all except zL, zS 
        key_list=list(sampler_list[0].columns)
        try:key_list.remove('zL');key_list.remove('zS')
        except: pass
    print('Keys:',key_list)
    print('NOTE: Need to change this so the bins cover the whole prior:')
    #N_chains = sampler_list[0].num_chains
    for s_i,sampler in enumerate(sampler_list):
        corner_samples = sampler[key_list]
        if plot_Ok:
            Ok = np.array([1-(sampler['OM']+sampler['Ode'])]).T.flatten()
            sampler['Ok'] = Ok
            print('Ok',Ok.shape,Ok)
            key_list.append('Ok')
        #if len(exclude_walker_list)>0: print(f'Excluding {len(exclude_walker_list)} walkers from the plot')
        #corner_samples=np.delete(corner_samples,exclude_walker_list,axis=1)
        if s_i == len(sampler_list)-1: truths=[truth_dict[k] for k in key_list]
        else: truths=None
        if fig is None: 
            fig,ax = pl.subplots(len(key_list),len(key_list),figsize=(2.2*len(key_list),2.2*len(key_list)))
        truth_list = [truth_dict[k_i] for k_i in key_list]
        print('TRUTH',truth_list)
        print('Nans and Inf:',np.sum(np.isnan(corner_samples)).to_numpy(),np.sum(np.isinf(corner_samples).to_numpy()))
        corner.corner(corner_samples,
            #truths=truth_list,
            #truth_color='k',
            labels=[label_dict[k_i] for k_i in key_list],
            fig=fig,
            color=color_list[s_i],
            range=[range_dict[k_i] for k_i in key_list],
            hist_kwargs={'density':True,'alpha':alpha_hist1d},
            hist2d_kwargs={'label':'_nolegend_','alpha':alpha_hist2d},
            contour_kwargs={'alpha':alpha_hist2d},
            label_kwargs={'fontsize':21},
            plot_datapoints=False)#,show_titles=True)
        for p_i in range(np.array(corner_samples).shape[1]):
            if add_text:
                ax[p_i,p_i].text(1,1,percentile_str(np.array(corner_samples)[:,p_i]),
                        horizontalalignment='right',
                        verticalalignment='top',
                        color='darkred',
                        transform=ax[p_i,p_i].transAxes,
                        fontsize=10)
            y_frac = 0.1+0.5*s_i/len(sampler_list)
            plot_mu_sig(ax[p_i,p_i],np.array(corner_samples)[:,p_i],c=color_list[s_i],y_frac = y_frac)
    if len(hist_ylim)!=0:
        for ii in range(len(ax)):
            ax_ymax = np.max([ax[ii,ii].get_ylim()[1],hist_ylim[ii][1]])
            ax[ii,ii].set_ylim(top=ax_ymax)
    for jj in range(len(ax)):
        ax[jj,jj].plot([truth_dict[key_list[jj]]]*2,ax[jj,jj].get_ylim(),c='k')
        for ii in range(jj):
            ax[jj,ii].set_xlim(range_dict[key_list[ii]])
            ax[jj,ii].plot([truth_dict[key_list[ii]]]*2,ax[jj,ii].get_ylim(),c='k',label='_nolegend_')
            ax[jj,ii].plot(ax[jj,ii].get_xlim(),[truth_dict[key_list[jj]]]*2,c='k',label='_nolegend_')
            ax[jj,ii].scatter([truth_dict[key_list[ii]]],[truth_dict[key_list[jj]]],s=50,c='k',label='_nolegend_',zorder=3)
    pl.tight_layout()     
    if len(legend_list)!=0: print(legend_list);ax[0,0].legend(legend_list,fontsize=8)
    if plot_Ok: print("Note: Ok is not an independent variable, so shouldn't be plotted as if it is.")
    return fig,ax