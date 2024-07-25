from scipy.stats import beta
from numpyro import distributions as dist 
import numpy as np

class beta_class:
    def __init__(self,mean=None,sigma=None,A=None,B=None):
        self.mean=mean
        self.sigma=sigma
        self.A=A
        self.B=B
        if self.mean is not None and self.sigma is not None:
            if self.A is None: self.A = self.mean*((self.mean*(1-self.mean)/self.sigma**2)-1)
            if self.B is None: self.B = (1-self.mean)*((self.mean*(1-self.mean)/self.sigma**2)-1) 

    def beta_func_scipy(self):
        assert self.A>0;assert self.B>0
        assert self.sigma**2<(self.mean*(1-self.mean))
        # print('A,B',self.A,self.B)
        return beta(a=self.A,b=self.B)

    def beta_func_numpyro(self,X):
        return np.exp(dist.Beta(self.A,self.B).log_prob(X))

    def max_sigma_for_unimodal_beta(self,X):
        try:
            len(X)
            return np.array([(elem*(1-elem)*np.min([elem/(1+elem),(1-elem)/(2-elem)]))**0.5 for elem in X])
        except:
            return (X*(1-X)*np.min([X/(1+X),(1-X)/(2-X)]))**0.5

    def min_sigma_for_bimodal_beta(self,X):
        try:
            len(X)
            return np.array([np.sqrt(elem*(1-elem)/2) for elem in X])
        except:
            return np.sqrt(X*(1-X)/2)
        

    def max_possible_sigma(self,X):
        try:
            len(X)
            return np.array([np.sqrt(elem*(1-elem)) for elem in X])
        except:
            return np.sqrt(X*(1-X))