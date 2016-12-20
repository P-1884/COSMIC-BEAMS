import numpy as np
import matplotlib.pyplot as plt
import contour_plot
import gelman_rubin



def triangle_plot(chain,params=[],labels=[],true_vals=[],best_params=[],smooth=5e3):
    """
        Plots the triangle plot for a sampled chain.
        chain = Input chain
        params = List of indices of parameters, otherwise every column of chain is used
        labels = Labels for parameters
        true_vales = If provided, plots the true values on the histograms and contours
        best_params = List of lists for each parameter (mean, minus uncertainty, plus uncertainty) plotted on histograms
        smooth = Smoothing scale for the contours. Contour will raise warning is this is too small. Set to 0 for no smoothing.
    """
    fntsz=18
    if len(params)==0:
        #If a list of parameter indices is not explicitly given, assume we plot all columns of chain except the last
        # (assumed to be likelihood)
        params=range(len(chain[0,:-1]))
    if len(labels)==0:
        labels=['%d' %i for i in range(len(params))]


    for i in range(len(params)):
        plt.subplot(len(params),len(params),i*(len(params)+1)+1)
        #PLot the histograms
        plt.hist(chain[:,params[i]],25,facecolor='#a3c0f6')
        if len(true_vals)!=0:
            plt.plot([true_vals[i],true_vals[i]],plt.gca().get_ylim(),'k',lw=2.5)
        if len(best_params)!=0:
            plt.plot([best_params[i][0],best_params[i][0]],plt.gca().get_ylim(),'r',lw=2.5)
            plt.plot([best_params[i][0]+best_params[i][2],best_params[i][0]+best_params[i][2]],plt.gca().get_ylim(),'r--',lw=2.5)
            plt.plot([best_params[i][0]-best_params[i][1],best_params[i][0]-best_params[i][1]],plt.gca().get_ylim(),'r--',lw=2.5)
        plt.ticklabel_format(style='sci',scilimits=(-3,5))

        #Plot the contours
        for j in range(0,i):
            plt.subplot(len(params),len(params),i*(len(params))+j+1)
            contour_plot.contour(chain,[params[j],params[i]],smooth=smooth)
            if len(true_vals)!=0:
                plt.plot([true_vals[j]],[true_vals[i]],'*k',markersize=10)
            plt.ticklabel_format(style='sci',scilimits=(-3,5))
        plt.ticklabel_format()

    for i in range(len(params)):
        ax=plt.subplot(len(params),len(params),len(params)*(len(params)-1)+i+1)
        ax.set_xlabel(labels[i],fontsize=fntsz)
        ax=plt.subplot(len(params),len(params),i*len(params)+1)
        ax.set_ylabel(labels[i],fontsize=fntsz)


####Example code####

### Load chains ###
#
# rt=''
# burn=1000
# chain0=np.loadtxt(rt+'testHMC-2sources-numeric.chain0')[burn::2,1:]
# chain1=np.loadtxt(rt+'testHMC-2sources-numeric.chain1')[burn::2,1:]
# chain2=np.loadtxt(rt+'testHMC-2sources-numeric.chain2')[burn::2,1:]
#
# ### Test convergence with gelman rubin ###
#
# step=gelman_rubin.converge_from_list([chain0,chain1,chain2],jump=500)
# if step!=1:
#     print 'Chain converged within',step,'steps'
# else:
#     print 'Chain unconverged'
#
#
# ### Compute best fitting parameters and uncertainties using percentiles ###
# best_prms = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
#                              zip(*np.percentile(chain0, [16, 50, 84],axis=0)))
# #Taken from emcee example
#
#
# ### Plot chain ###
# labels=['ex_1', 'ex_2', 'ey_1', 'ey_2', 's_1', 's_2']
# labels=['$%s$' %l for l in labels]
# params=range(6)
# true_vals=[5.865E-02,5.631E-01,-2.973E-02,2.954E-01,3.592E-06,7.746E-06]
#
# triangle_plot(chain0,params,labels,true_vals=true_vals,best_params=best_prms,smooth=6e3)
# plt.show()