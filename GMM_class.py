import numpy as np
from scipy.stats import norm,truncnorm
from sklearn.mixture import GaussianMixture as GMM
import matplotlib.pyplot as pl

class GMM_class:
    def __init__(self,list_of_mu=None,list_of_sigma=None,list_of_weights=None):
        self.list_of_mu = np.array(list_of_mu)
        self.list_of_sigma = np.array(list_of_sigma)
        self.list_of_weights = np.array(list_of_weights)

    def sample(self,N_samples,trunc_at_zero):
        print('NOTE - AM ROUNDING PROBABILITIES - SHOULD REMOVE THIS')
        p = np.round(self.list_of_weights,6)
        component = np.random.choice(np.arange(len(self.list_of_weights)),p = p,size=N_samples).astype('int')
        # print(component)
        # print(self.list_of_mu[component])
        # print(self.list_of_sigma[component])
        # print(N_samples)
        # print(np.array(self.list_of_mu[component])/np.array(self.list_of_sigma[component]))
        if trunc_at_zero:
            return truncnorm(a=-np.array(self.list_of_mu[component])/np.array(self.list_of_sigma[component]),
                            b=np.inf,
                            loc=self.list_of_mu[component],
                            scale=self.list_of_sigma[component]).rvs(size=N_samples)
        else:
            return np.random.normal(loc=self.list_of_mu[component],scale=self.list_of_sigma[component],size=N_samples)

    def fit(self,data,n_components=2):
        GMM_fit =  GMM(n_components=n_components).fit(np.array(data))
        self.list_of_mu = GMM_fit.means_.flatten()
        self.list_of_sigma = np.sqrt(GMM_fit.covariances_.flatten())
        self.list_of_weights = GMM_fit.weights_
        print('Best fit values:',{'mu':self.list_of_mu,
                                  'sigma':self.list_of_sigma,
                                  'weights':self.list_of_weights})
        return self

    def plot(self,X_plot,trunc_at_zero,label_components=False,ax = None,label=None,
            alpha=1,linewidth=4,plot_components=True,color=None,component_color=None,legend=True,total_color='k'):
        if plot_components and component_color is None: component_color = color
        if ax is None: fig,ax = pl.subplots(figsize=(8,5))
        component_dict = {}
        for i in range(len(self.list_of_mu)):
            loc_i = self.list_of_mu[i]
            scale_i = self.list_of_sigma[i]
            w_i = self.list_of_weights[i]
            if trunc_at_zero:
                component_dict[f'Component_{i}'] = truncnorm(loc=loc_i,scale=scale_i,
                                                    a=-loc_i/scale_i,b=np.inf).pdf(X_plot)*w_i
            else:
                component_dict[f'Component_{i}'] = norm(loc=loc_i,scale=scale_i).pdf(X_plot)*w_i 
        if label is None: label = 'Total'
        ax.plot(X_plot,np.sum(np.array(list(component_dict.values())),axis=0),'--',linewidth=linewidth,
                label=label,c=total_color,alpha=alpha)
        if plot_components:
            for p_i in range(len(self.list_of_mu)):
                if label_components: label_0 = f'Component {p_i}'
                else: label_0=None
                ax.plot(X_plot,component_dict[f'Component_{p_i}'],'--',label=label_0,color=component_color)
        if legend: ax.legend()