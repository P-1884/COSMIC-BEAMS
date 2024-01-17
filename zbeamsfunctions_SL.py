# -*- coding: utf-8 -*-
"""
"""

import numpy as np
from scipy.integrate import quad
from scipy import interpolate
from astropy.cosmology import LambdaCDM,FlatLambdaCDM,wCDM,FlatwCDM,w0waCDM,Flatw0waCDM
from Lenstronomy_Cosmology import Background, LensCosmo
from scipy.stats import multivariate_normal
def r_SL(zL,zS,cosmo):
    LensCosmo_i = LensCosmo(z_lens=zL,z_source=zS,cosmo=cosmo)
    D_LS = LensCosmo_i.dds
    D_S = LensCosmo_i.ds
    r_value = D_LS.value/D_S.value
    if np.isnan(r_value).any():
        print('Redshift input',zL,zS,zL<zS)
        assert False
    return r_value

#Note: CANNOT just adapt the zBEAMS spectroscopic code to have larger errorbars as they assume z is known.
#E.g. See Eqn 4.4 of BEAMS paper.
def likelihood_SL(OM,Ode,H0,w,wa,zL,zS,r_obs,sigma_r_obs,cosmo_type):
    '''
    If this code turns out to be quite slow, the previous likelihood function used interpolation.
    '''
    if cosmo_type == 'FlatLambdaCDM': cosmo = FlatLambdaCDM(H0=H0, Om0=OM)#; assert OM+Ode==1; assert w==-1
    if cosmo_type == 'LambdaCDM': cosmo = LambdaCDM(H0=H0, Om0=OM, Ode0=Ode)#; assert w==-1
    if cosmo_type == 'FlatwCDM': cosmo = Flatw0waCDM(H0=H0, Om0=OM, w0=w, wa=wa)#; assert OM+Ode==1 #Assert Flat cosmology.
    if cosmo_type == 'w0CDM': cosmo = wCDM(H0=H0, Om0=OM, Ode0=Ode, w0=w)
    if cosmo_type =='wCDM': cosmo = w0waCDM(H0=H0, Om0=OM, Ode0=Ode, w0=w, wa=wa)
    r_theory = r_SL(zL,zS,cosmo)   
    chi2 = ((r_obs-r_theory)/sigma_r_obs)**2
    likeli = -0.5*sum(chi2)
    return likeli

def likelihood_spec_contam_SL(OM,Ode,H0,w,wa,zL,zS,r_obs,sigma_r_obs_1,sigma_r_obs_2,P_tau,cosmo_type):
    '''
    If this code turns out to be quite slow, the previous likelihood function used interpolation.
    '''
    if cosmo_type == 'FlatLambdaCDM': cosmo = FlatLambdaCDM(H0=H0, Om0=OM)#; assert OM+Ode==1; assert w==-1
    if cosmo_type == 'LambdaCDM': cosmo = LambdaCDM(H0=H0, Om0=OM, Ode0=Ode)#; assert w==-1
    if cosmo_type == 'FlatwCDM': cosmo = Flatw0waCDM(H0=H0, Om0=OM, w0=w, wa=wa)#; assert OM+Ode==1 #Assert Flat cosmology.
    if cosmo_type == 'w0CDM': cosmo = wCDM(H0=H0, Om0=OM, Ode0=Ode, w0=w)
    if cosmo_type =='wCDM': cosmo = w0waCDM(H0=H0, Om0=OM, Ode0=Ode, w0=w, wa=wa)
    r_theory = r_SL(zL,zS,cosmo)   

    ###
    '''
    Gamma: What supernovae host galaxy is used
    Tau: Is it a 1A supernovae or not.
    sigmuo => Distance modulus, assuming it is a 1A
    sigmuo2 => Distance modulus, assuming it is not a 1A => should be large (=> uninformative).
    mu_theory1 => Theoretical distance modulus, based on redshifts of host-galaxy 1 (with prob P_gamma)
    mu_theory2 => Theoretical distance modulus, based on redshifts of host-galaxy 2 (with prob 1-P_gamma)
    For the Lensing case, I can get rid of P_gamma, as we don't have ambiguous host galaxies.
    '''
    chi2_1 = ((r_obs-r_theory)/sigma_r_obs_1)**2
    chi2_3 = ((r_obs-r_theory)/sigma_r_obs_2)**2
    ###
    #Adapted from Eqn 4.4 & 4.5 of zBEAMS:
    L1 = P_tau*((1/np.sqrt(2*np.pi*sigma_r_obs_1**2))*np.exp(-chi2_1/2))
    L3 = (1-P_tau)*((1/np.sqrt(2*np.pi*sigma_r_obs_2**2))*np.exp(-chi2_3/2))
    ###
    return np.sum(np.log(L1 + L3))

def multivariate_normal_func(x,mean,cov):
    try:
        mvn = multivariate_normal(mean=mean,cov=cov).pdf(x)
    except:
        print('mean',mean,'cov',cov)
        assert False
    if np.isnan(mvn): print('mean',mean,'cov',cov); assert False 
    return mvn

def likelihood_phot_contam_SL(OM,Ode,H0,w,wa, #Cosmological parameters to constrain
                    mu_zL_g_L,mu_zS_g_L,mu_zL_g_NL,mu_zS_g_NL,#Mean values of p(z|tau) multi-variate gaussian (g_ = 'given')
                    si_00_g_L,si_11_g_L, #Covariance values of p(z|Lens) - **Assuming diagonal**
                    si_00_g_NL,si_11_g_NL, #Covariance values of p(z|Not-Lens) - **Assuming diagonal**
                    zL,zS,#Redshifts to constrain
                    zL_obs,zS_obs,sigma_zL_obs,sigma_zS_obs,r_obs,sigma_r_obs_1,sigma_r_obs_2,P_tau,cosmo_type):
    '''
    NOTE: Am having to assume diagonal covariance matrix for P(z|Tau) unless I can find a quick way of vectorising scipy.stats.multivariate_normal
    If this code turns out to be quite slow, the previous likelihood function used interpolation.
    '''
    if cosmo_type == 'FlatLambdaCDM': cosmo = FlatLambdaCDM(H0=H0, Om0=OM)#; assert OM+Ode==1; assert w==-1
    if cosmo_type == 'LambdaCDM': cosmo = LambdaCDM(H0=H0, Om0=OM, Ode0=Ode)#; assert w==-1
    if cosmo_type == 'FlatwCDM': cosmo = Flatw0waCDM(H0=H0, Om0=OM, w0=w, wa=wa)#; assert OM+Ode==1 #Assert Flat cosmology.
    if cosmo_type == 'w0CDM': cosmo = wCDM(H0=H0, Om0=OM, Ode0=Ode, w0=w)
    if cosmo_type =='wCDM': cosmo = w0waCDM(H0=H0, Om0=OM, Ode0=Ode, w0=w, wa=wa)
    r_theory = r_SL(zL,zS,cosmo)   

    ###
    '''
    Gamma: What supernovae host galaxy is used
    Tau: Is it a 1A supernovae or not.
    sigmuo => Distance modulus, assuming it is a 1A
    sigmuo2 => Distance modulus, assuming it is not a 1A => should be large (=> uninformative).
    mu_theory1 => Theoretical distance modulus, based on redshifts of host-galaxy 1 (with prob P_gamma)
    mu_theory2 => Theoretical distance modulus, based on redshifts of host-galaxy 2 (with prob 1-P_gamma)
    For the Lensing case, I can get rid of P_gamma, as we don't have ambiguous host galaxies.
    '''
    #P(D|tau,z,r) (also added in redshift uncertainty below):
    chi2_1 = ((r_obs-r_theory)/sigma_r_obs_1)**2 #if it is a lens
    chi2_3 = ((r_obs-r_theory)/sigma_r_obs_2)**2 #if it is not a lens
    #P(z|tau):
    chi_z_g_L = list(map(multivariate_normal_func,np.array([zL,zS]).T, #MCMC proposals of zL and zS
                         (np.array([[mu_zL_g_L]*len(zL),[mu_zS_g_L]*len(zL)]).T), #Mu of P(z|L)
                                                    [np.diag([si_00_g_L,si_11_g_L])]*len(zL))) #Cov of P(z|NL), assume diagonal
    chi_z_g_NL = list(map(multivariate_normal_func, np.array([zL,zS]).T, #MCMC proposals of zL and zS
                         (np.array([[mu_zL_g_NL]*len(zL),[mu_zS_g_NL]*len(zL)]).T), #Mu of P(z|NL)
                                                    [np.diag([si_00_g_NL,si_11_g_NL])]*len(zL))) #Cov of P(z|NL), assume diagonal
    ###
    #Adapted from Eqn 4.4 & 4.5 of zBEAMS:
    L1 = P_tau*((1/np.sqrt(2*np.pi*sigma_r_obs_1**2))*np.exp(-chi2_1/2))
    L3 = (1-P_tau)*((1/np.sqrt(2*np.pi*sigma_r_obs_2**2))*np.exp(-chi2_3/2))
    ##
    L4_L = (1/np.sqrt(2*np.pi*sigma_zL_obs**2))*np.exp(-0.5*((zL-zL_obs)/sigma_zL_obs)**2)
    L4_S = (1/np.sqrt(2*np.pi*sigma_zS_obs**2))*np.exp(-0.5*((zS-zS_obs)/sigma_zS_obs)**2)
    ##
    L5 = chi_z_g_L
    L6 = chi_z_g_NL
    ###
    likelihood_i = (L1*chi_z_g_L + L3*chi_z_g_NL)*L4_L*L4_S
    if (likelihood_i==0).any():
        print('Zero likelihood')
        '''
        print('L1',L1,'P_tau',P_tau,'sigma_r_obs_1',sigma_r_obs_1,'chi2_1',chi2_1,
              'factor',((1/np.sqrt(2*np.pi*sigma_r_obs_1**2)),'exp',np.exp(-chi2_1/2)))
        print('L3',L3,'1-P_tau',1-P_tau,'sigma_r_obs_2',sigma_r_obs_2,'chi2_3',chi2_3,
              'factor',((1/np.sqrt(2*np.pi*sigma_r_obs_2**2)),'exp',np.exp(-chi2_3/2)))
        assert False'''
    L_tot = np.sum(np.log(likelihood_i))
    return L_tot 


def likelihood_phot_SL(OM,Ode,H0,w,wa, #Cosmological parameters to constrain
                        mu_zL_g_L,mu_zS_g_L,#Mean values of p(z|tau) multi-variate gaussian (g_ = 'given')
                        si_00_g_L,si_11_g_L, #Covariance values of p(z|Lens) - **Assuming diagonal**
                        zL,zS,#Redshifts to constrain
                        zL_obs,zS_obs,sigma_zL_obs,sigma_zS_obs,r_obs,sigma_r_obs_1,cosmo_type):
    '''
    NOTE: Am having to assume diagonal covariance matrix for P(z|Tau) unless I can find a quick way of vectorising scipy.stats.multivariate_normal
    If this code turns out to be quite slow, the previous likelihood function used interpolation.
    '''
    if cosmo_type == 'FlatLambdaCDM': cosmo = FlatLambdaCDM(H0=H0, Om0=OM)#; assert OM+Ode==1; assert w==-1
    if cosmo_type == 'LambdaCDM': cosmo = LambdaCDM(H0=H0, Om0=OM, Ode0=Ode)#; assert w==-1
    if cosmo_type == 'FlatwCDM': cosmo = Flatw0waCDM(H0=H0, Om0=OM, w0=w, wa=wa)#; assert OM+Ode==1 #Assert Flat cosmology.
    if cosmo_type == 'w0CDM': cosmo = wCDM(H0=H0, Om0=OM, Ode0=Ode, w0=w)
    if cosmo_type =='wCDM': cosmo = w0waCDM(H0=H0, Om0=OM, Ode0=Ode, w0=w, wa=wa)
    r_theory = r_SL(zL,zS,cosmo)   
    print('r_theory',r_theory)
    ###
    '''
    Gamma: What supernovae host galaxy is used
    Tau: Is it a 1A supernovae or not.
    sigmuo => Distance modulus, assuming it is a 1A
    sigmuo2 => Distance modulus, assuming it is not a 1A => should be large (=> uninformative).
    mu_theory1 => Theoretical distance modulus, based on redshifts of host-galaxy 1 (with prob P_gamma)
    mu_theory2 => Theoretical distance modulus, based on redshifts of host-galaxy 2 (with prob 1-P_gamma)
    For the Lensing case, I can get rid of P_gamma, as we don't have ambiguous host galaxies.
    '''
    #P(D|tau,z,r) (also added in redshift uncertainty below):
    chi2_1 = ((r_obs-r_theory)/sigma_r_obs_1)**2 #if it is a lens
    #P(z|tau):
    chi_z_g_L = list(map(multivariate_normal_func,np.array([zL,zS]).T, #MCMC proposals of zL and zS
                          np.array([[mu_zL_g_L]*len(zL),
                                    [mu_zS_g_L]*len(zL)]).T, #Mu of P(z|L)
                         [np.diag([si_00_g_L,si_11_g_L])]*len(zL))) #Cov of P(z|L), assume diagonal
    ###
    #Adapted from Eqn 4.4 & 4.5 of zBEAMS:
    L1 = ((1/np.sqrt(2*np.pi*sigma_r_obs_1**2))*np.exp(-chi2_1/2))
    ##
    L4_L = (1/np.sqrt(2*np.pi*sigma_zL_obs**2))*np.exp(-0.5*((zL-zL_obs)/sigma_zL_obs)**2)
    L4_S = (1/np.sqrt(2*np.pi*sigma_zS_obs**2))*np.exp(-0.5*((zS-zS_obs)/sigma_zS_obs)**2)
    ##
    L5 = chi_z_g_L
    ###
    likelihood_i = (L1*chi_z_g_L)*L4_L*L4_S
    if (likelihood_i==0).any():
        print('Zero likelihood')
        '''
        print('L1',L1,'sigma_r_obs_1',sigma_r_obs_1,'chi2_1',chi2_1,
              'factor',((1/np.sqrt(2*np.pi*sigma_r_obs_1**2)),'exp',np.exp(-chi2_1/2)))
        assert False'''
    L_tot = np.sum(np.log(likelihood_i))
    return L_tot 

def mu_w(z,OM,H0,w):
    def H(z,OM,H0,w):
        if w==-1:
            #Eqn 2.3 of zBEAMS paper, assuming O_k = 0 and w=-1.
            return H0*np.sqrt(OM*(1+z)**3+(1-OM))
        else:
            #Eqn 2.3 of zBEAMS paper, assuming O_k = 0.
            return H0*np.sqrt(OM*(1+z)**3+(1-OM)*(1+z)**(3*(w+1)))
            
    def dL(z,OM,H0,w):
        c=2.99792e5 #c in km/s
        #Luminosity distance for a flat universe (O_k=0) - see here https://en.wikipedia.org/wiki/Distance_measure
        return (1+z)*quad(lambda x:c/H(x,OM,H0,w),0,z)[0]
    #NOTE, THIS SEEMS TO BE IN THE WRONG UNITS - SHOULD BE IN Mpc (See Eqn 2.1 of zBEAMS)?
    return 5*np.log10(dL(z,OM,H0,w))+25


def likelihood_phot(z,muo,sigmuo,OM,H0,w,zo,sigzo,b):
    def mu_interp(z,OM,H0,w):
        assert False #Need to fix this to allow for redshift-evolving dark-matter EoS
        def H(z,OM,H0,w):
            assert False #Need to fix this to allow for redshift-evolving dark-matter EoS
            if w==-1:
                return H0*np.sqrt(OM*(1+z)**3+(1-OM))
            else:
                return H0*np.sqrt(OM*(1+z)**3+(1-OM)*(1+z)**(3*(w+1)))
                
        def dL(z,OM,H0,w):
            assert False #Need to fix this to allow for redshift-evolving dark-matter EoS
            c=2.99792e5
            return (1+z)*quad(lambda x:c/H(x,OM,H0,w),0,z)[0]
            
        def mu_w(z,OM,H0,w):
            assert False #Need to fix this to allow for redshift-evolving dark-matter EoS
            return 5*np.log10(dL(z,OM,H0,w))+25
        #The purpose of np.vectorize is to transform functions which are not numpy-aware (e.g. take floats
        #as input and return floats as output) into functions that can operate on (and return) numpy arrays.        
        mu_w_vectorized = np.vectorize(mu_w)
                
        z_spl = np.linspace(np.min(z),np.max(z),50)
        mu_spl = mu_w_vectorized(z_spl,OM,H0,w)
        tck = interpolate.splrep(z_spl, mu_spl)
        mu_int = interpolate.splev(z, tck)
        return mu_int
    
    mu_theory = mu_interp(z,OM,H0,w)
    '''
    Why do we not care about the 1/sqrt(2pi*sigma^2) here?
    '''
    chi2 = ((muo-mu_theory)/sigmuo)**2
    chi3 = ((z-zo)/sigzo)**2
    chi4 = np.log(z) - b*z
    likeli = -0.5*sum(chi2)
    prior = -0.5*sum(chi3)
    priorz = sum(chi4)
    return likeli + prior + priorz
        

def likelihood_spec(z,z2,muo,sigmuo,sigmuo2,P_gamma,P_tau,offset,OM,H0,w):
    def mu_interp(z,OM,H0,w):
        def H(z,OM,H0,w):
            assert False #Need to fix this to allow for redshift-evolving dark-matter EoS
            if w==-1:
                return H0*np.sqrt(OM*(1+z)**3+(1-OM))
            else:
                return H0*np.sqrt(OM*(1+z)**3+(1-OM)*(1+z)**(3*(w+1)))
                
        def dL(z,OM,H0,w):
            assert False #Need to fix this to allow for redshift-evolving dark-matter EoS
            c=2.99792e5
            return (1+z)*quad(lambda x:c/H(x,OM,H0,w),0,z)[0]
            
        def mu_w(z,OM,H0,w):
            assert False #Need to fix this to allow for redshift-evolving dark-matter EoS
            return 5*np.log10(dL(z,OM,H0,w))+25
                
        mu_w_vectorized = np.vectorize(mu_w)
                
        z_spl = np.linspace(np.min(z),np.max(z),50)
        mu_spl = mu_w_vectorized(z_spl,OM,H0,w)
        tck = interpolate.splrep(z_spl, mu_spl)
        mu_int = interpolate.splev(z, tck)
        return mu_int
    '''
    Gamma: What supernovae host galaxy is used
    Tau: Is it a 1A supernovae or not.
    sigmuo => Distance modulus, assuming it is a 1A
    sigmuo2 => Distance modulus, assuming it is not a 1A => should be large (=> uninformative).
    mu_theory1 => Theoretical distance modulus, based on redshifts of host-galaxy 1 (with prob P_gamma)
    mu_theory2 => Theoretical distance modulus, based on redshifts of host-galaxy 2 (with prob 1-P_gamma)
    For the Lensing case, I can get rid of P_gamma, as we don't have ambiguous host galaxies.
    '''
    mu_theory1 = mu_interp(z,OM,H0,w)
    mu_theory2 = mu_interp(z2,OM,H0,w)
    chi2_1 = ((muo-mu_theory1)/sigmuo)**2
    chi2_2 = ((muo-mu_theory2)/sigmuo)**2
    chi2_3 = ((muo-mu_theory1)/sigmuo2)**2
    chi2_4 = ((muo-mu_theory2)/sigmuo2)**2
    
    L1 = P_tau*P_gamma*((1/np.sqrt(2*np.pi*sigmuo**2))*np.exp(-chi2_1/2))
    L2 = P_tau*(1-P_gamma)*((1/np.sqrt(2*np.pi*sigmuo**2))*np.exp(-chi2_2/2))
    L3 = (1-P_tau)*(P_gamma)*((1/np.sqrt(2*np.pi*sigmuo2**2))*np.exp(-chi2_3/2))
    L4 = (1-P_tau)*(1-P_gamma)*((1/np.sqrt(2*np.pi*sigmuo2**2))*np.exp(-chi2_4/2))
    return np.sum(np.log(L1 + L2 + L3 + L4))


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
        print('SUM',tot,tot95,tot68)
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
