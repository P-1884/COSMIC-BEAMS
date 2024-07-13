import matplotlib.pyplot as pl
import pandas as pd
import numpy as np

def corner_scatter_plot(df_list,legend_list=None,columns_to_plot = None,label_dict = None,range_dict = {},suptitle=None,color_list = [],
                        alpha = 0.5,fontsize=18,linewidth=3,bin_dict = None,s = None):
    if isinstance(alpha,float): alpha = [alpha for ii in range(len(df_list))]
    if len(color_list)==0: color_list = pl.rcParams['axes.prop_cycle'].by_key()['color']
    if legend_list is None: legend_list = [None for ii in range(len(df_list))]
    if columns_to_plot is None: columns_to_plot = df_list[0].columns
    fig,ax = pl.subplots(len(columns_to_plot),len(columns_to_plot),figsize=(5*len(columns_to_plot),5*len(columns_to_plot)))
    for n_i,df_i in enumerate(df_list):
        for ii in range(len(columns_to_plot)):
            try: 
                x_min = range_dict[columns_to_plot[ii]][0]
                x_max = range_dict[columns_to_plot[ii]][1]
            except:
                x_min = np.min([np.min(df_list[d_i][columns_to_plot[ii]]) for d_i in range(len(df_list))])
                x_max = np.max([np.max(df_list[d_i][columns_to_plot[ii]]) for d_i in range(len(df_list))]) 
            if bin_dict is None:
                bins = np.linspace(x_min,x_max,20)
            else:
                bins = bin_dict[columns_to_plot[ii]]
            ax[ii,ii].hist(df_i[columns_to_plot[ii]],label=legend_list[n_i],fill=False,bins=bins,density=True,alpha=alpha[n_i],
                            edgecolor=color_list[n_i],linewidth=linewidth)
            if label_dict is None: label_i = columns_to_plot[ii]
            else: label_dict = label_dict[columns_to_plot[ii]]
            ax[ii,ii].set_title(label_i,fontsize=fontsize+3,fontweight='bold')
            ax[ii,ii].set_xlabel(label_i,fontsize=fontsize)
            ax[ii,ii].legend(fontsize=fontsize-3)#,loc='upper left')
            # if (df_i[columns_to_plot[ii]]>x_max).any():
            #     print('Right',columns_to_plot[ii])
            #     X = x_min + 0.8*(x_max-x_min)
            #     Y = 1+n_i
            #     ylim = ax[ii,ii].get_ylim()
            #     ax[ii,ii].arrow(X,Y,dx=0.1*(x_max-x_min),dy=0,width=0.01,head_width=0.05*(ylim[1]-ylim[0]),
            #         color=color_list[n_i],label='_nolegend_')
            # if (df_i[columns_to_plot[ii]]<x_min).any():
            #     print('Left',columns_to_plot[ii])
            #     X = x_min + 0.2*(x_max-x_min)
            #     Y = 1+n_i
            #     ylim = ax[ii,ii].get_ylim()
            #     ax[ii,ii].arrow(X,Y,dx=-0.1*(x_max-x_min),dy=0,width=0.01,head_width=0.05*(ylim[1]-ylim[0]),
            #         color=color_list[n_i],label='_nolegend_')
        for x in range(len(columns_to_plot)):
            for y in range(len(columns_to_plot)):
                if x>y: 
                    try: fig.delaxes(ax[y,x])
                    except: continue #May have already been deleted
                if x>=y: continue
                ax[y,x].scatter(df_i[columns_to_plot[x]],df_i[columns_to_plot[y]],alpha=alpha[n_i],color=color_list[n_i],s=s)
                if y==len(columns_to_plot)-1:
                    if label_dict is not None: label_x = label_dict[columns_to_plot[x]]
                    else: label_x = columns_to_plot[x]
                    ax[y,x].set_xlabel(label_x,fontsize=fontsize)
                if x==0: 
                    if label_dict is not None: label_y = label_dict[columns_to_plot[y]]
                    else: label_y = columns_to_plot[y]
                    ax[y,x].set_ylabel(label_y,fontsize=fontsize)
    for x in range(len(columns_to_plot)):
        try: ax[x,x].set_xlim(range_dict[columns_to_plot[x]])
        except: pass
        for y in range(len(columns_to_plot)):  
            ax[y,x].set_xlim(ax[x,x].get_xlim())
            if x!=y: ax[y,x].set_ylim(ax[y,y].get_xlim())
    pl.tight_layout()
    if suptitle is not None: pl.suptitle(suptitle,fontsize=fontsize+8,fontweight='bold')
    pl.show()
