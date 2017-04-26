# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 12:06:21 2017

@author: ethan
"""

import numpy as np
from scipy.integrate import quad
from scipy import interpolate


def mu_w(z,OM,H0,w):
    def H(z,OM,H0,w):
        if w==-1:
            return H0*np.sqrt(OM*(1+z)**3+(1-OM))
        else:
            return H0*np.sqrt(OM*(1+z)**3+(1-OM)*(1+z)**(3*(w+1)))
            
    def dL(z,OM,H0,w):
        c=2.99792e5
        return (1+z)*quad(lambda x:c/H(x,OM,H0,w),0,z)[0]
    
    return 5*np.log10(dL(z,OM,H0,w))+25


def likelihood_phot(z,muo,sigmuo,OM,H0,w,zo,sigzo,b):
    def mu_interp(z,OM,H0,w):
        def H(z,OM,H0,w):
            if w==-1:
                return H0*np.sqrt(OM*(1+z)**3+(1-OM))
            else:
                return H0*np.sqrt(OM*(1+z)**3+(1-OM)*(1+z)**(3*(w+1)))
                
        def dL(z,OM,H0,w):
            c=2.99792e5
            return (1+z)*quad(lambda x:c/H(x,OM,H0,w),0,z)[0]
            
        def mu_w(z,OM,H0,w):
            return 5*np.log10(dL(z,OM,H0,w))+25
                
        mu_w_vectorized = np.vectorize(mu_w)
                
        z_spl = np.linspace(np.min(z),np.max(z),50)
        mu_spl = mu_w_vectorized(z_spl,OM,H0,w)
        tck = interpolate.splrep(z_spl, mu_spl)
        mu_int = interpolate.splev(z, tck)
        return mu_int
    
    mu_theory = mu_interp(z,OM,H0,w)
    chi2 = ((muo-mu_theory)/sigmuo)**2
    chi3 = ((z-zo)/sigzo)**2
    chi4 = np.log(z) - b*z
    likeli = -0.5*sum(chi2)
    prior = -0.5*sum(chi3)
    priorz = sum(chi4)
    return likeli + prior + priorz
        

def likelihood(z,muo,sigmuo,OM,H0,w,zo,sigzo):
    def mu_interp(z,OM,H0,w):
        def H(z,OM,H0,w):
            if w==-1:
                return H0*np.sqrt(OM*(1+z)**3+(1-OM))
            else:
                return H0*np.sqrt(OM*(1+z)**3+(1-OM)*(1+z)**(3*(w+1)))
                
        def dL(z,OM,H0,w):
            c=2.99792e5
            return (1+z)*quad(lambda x:c/H(x,OM,H0,w),0,z)[0]
            
        def mu_w(z,OM,H0,w):
            return 5*np.log10(dL(z,OM,H0,w))+25
                
        mu_w_vectorized = np.vectorize(mu_w)
                
        z_spl = np.linspace(np.min(z),np.max(z),50)
        mu_spl = mu_w_vectorized(z_spl,OM,H0,w)
        tck = interpolate.splrep(z_spl, mu_spl)
        mu_int = interpolate.splev(z, tck)
        return mu_int
    
    mu_theory = mu_interp(z,OM,H0,w)
    chi2 = ((muo-mu_theory)/sigmuo)**2
    chi3 = ((z-zo)/sigzo)**2
    likeli = -0.5*sum(chi2)
    prior = -0.5*sum(chi3)
    return likeli + prior


def contour(chain,p,**kwargs):
    def findconfidence(H):
        H2 = H.ravel()
        H2 = np.sort(H2)
        
        #Cut out the very low end
        #H2 = H2[H2>100]
        
        #Loop through this flattened array until we find the value in the bin 
        #which contains 95% of the points
        tot = sum(H2)
        tot95=0
        tot68=0
        
        #Changed this to 68% and 30% C.I
        for i in range(len(H2)):
            tot95 += H2[i]
            if tot95 >= 0.05*tot:
                N95 = H2[i]
                #print i
                break
            
        for i in range(len(H2)):
            tot68 += H2[i]
            if tot68>=0.32*tot:
                N68 = H2[i]
                break   
        return max(H2),N95,N68
    
    binsize=50
    H, xedges, yedges = np.histogram2d(chain[:,p[0]],chain[:,p[1]], bins=(binsize,binsize))
    
    x=[]
    y=[]
    z=[]
    for i in range(len(xedges[:-1])):
        for j in range(len(yedges[:-1])):
            x.append(xedges[:-1][i])
            y.append(yedges[:-1][j])
            z.append(H[i, j])

    if 'smooth' in kwargs:
        SMOOTH=True
        smth=kwargs['smooth']
        if smth==0:
            SMOOTH=False
    else:
        SMOOTH=True
        smth=10e5
    if SMOOTH:
        sz=50
        spl = interpolate.bisplrep(x, y, z,  s=smth)
        X = np.linspace(min(xedges[:-1]), max(xedges[:-1]), sz)
        Y = np.linspace(min(yedges[:-1]), max(yedges[:-1]), sz)
        Z = interpolate.bisplev(X, Y, spl)
    else:
        X=xedges[:-1]
        Y=yedges[:-1]
        Z=H
    
    #I think this is the weird thing I have to do to make the contours work properly
    X1=np.zeros([len(X), len(X)])
    Y1=np.zeros([len(X), len(X)])
    for i in range(len(X)):
        X1[ :, i]=X
        Y1[i, :]=Y
    X=X1
    Y=Y1
    
    N100,N95,N68 = findconfidence(Z)
    
    return N100,N95,N68,X,Y,Z
