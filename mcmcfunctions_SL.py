#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  7 22:52:26 2017

@author: ethan
"""

import numpy as np
import random as rand
import sys
import emcee
import time
from multiprocessing import Pool

#This function needs to be global (i.e. not defined within a function), so the multiprocessing works:
def likelihood_with_bounds(MCMC_parameters, #Cosmology to find. NOTE If change the order of these, need to change the order in which 
                            #parameters in cur_state are called. 
        #MCMC_parameters order:
        #[0]:OM
        #[1]:OD
        #[2]:H0
        #[3]:W0
        #[4,5,6,7]: mu_zX_g_Y
        #[8,9,10,11]: si_X_g_Y  #Covariance values of p(z|Lens) - **Assuming diagonal**
        #[12:12+len(r_obs)]: ZL_MCMC
        #[12+len(r_obs):]: ZS_MCMC
        ZL_OBS=None,ZS_OBS=None,SIGMA_ZL_OBS=None,SIGMA_ZS_OBS=None,
        R_OBS=None,SIGMA_R_OBS=None, #Inputs
        likelihood=None, # Log-likelihood function, without any bounds
        SIGMA_R_OBS_2=None,P_TAU=None,# Optional Inputs if Contaminated=True
        COSMO_TYPE=None,FLAT_BOOL=None,COMPLEX_EOS_BOOL=None,CONTAMINATED=None,PHOTOMETRIC=None):
    #OM,OD,H0,W0
    if MCMC_parameters[0]<0 or MCMC_parameters[0]>1: print('Outside OM bound'); return -np.inf #OM
    if MCMC_parameters[1]<0 or MCMC_parameters[1]>1: print('Outside OD bound'); return -np.inf #OD
    if MCMC_parameters[2]>= 200 or MCMC_parameters[2] <= 10: print('Outside H0 bound');return -np.inf #H0
    if MCMC_parameters[3]<=-6 or MCMC_parameters[3]>4: print('Outside W0 bound');return -np.inf #W0
    '''
    Note, if the MCMC is going very slowly, there may be a way to speed it up in the flat and or w=-1 cases by removing
    them as arguments (rather than setting the likelihood=-np.inf if they are wrong):
    '''
    if FLAT_BOOL: 
        if MCMC_parameters[0]+MCMC_parameters[1]!=1: return -np.inf #Make sure Omega_K=0
    if not COMPLEX_EOS_BOOL:
        if MCMC_parameters[2]!=-1: return -np.inf #Make sure W0 is -1.
    if not CONTAMINATED:return likelihood(MCMC_parameters[0],MCMC_parameters[1],
                                            MCMC_parameters[2],MCMC_parameters[3],
                                            ZL_OBS,ZS_OBS,R_OBS,SIGMA_R_OBS,COSMO_TYPE)
    if CONTAMINATED and not PHOTOMETRIC: return likelihood(MCMC_parameters[0],MCMC_parameters[1],
                                        MCMC_parameters[2],MCMC_parameters[3],
                                        ZL_OBS,ZS_OBS,R_OBS,SIGMA_R_OBS,SIGMA_R_OBS_2,P_TAU,COSMO_TYPE)
    if PHOTOMETRIC:
        if (MCMC_parameters[12:12+len(ZL_OBS)]>MCMC_parameters[12+len(ZL_OBS):]).any():
            print('Redshifts unphysical')
            return -np.inf #Assert zL<zS
        if (MCMC_parameters[12:12+len(ZL_OBS)]<0).any() or (MCMC_parameters[12+len(ZL_OBS):]<0).any():
            print('Redshifts below zero')
            return -np.inf #Assert zL>0 and zS>0
        if (MCMC_parameters[4:12]<0).any():
            print('Redshift population priors below zero')
            return -np.inf #Assert mu and sigma are >0
        if (MCMC_parameters[8:12]>10).any():
            print('Redshift sigmas too extreme')
            return -np.inf #Assert redshift sigmas are <10.
    if CONTAMINATED and PHOTOMETRIC: return likelihood(
            MCMC_parameters[0],MCMC_parameters[1], #Cosmological parameters to constrain
            MCMC_parameters[2],MCMC_parameters[3], #Cosmological parameters to constrain
            #
            MCMC_parameters[4],MCMC_parameters[5],MCMC_parameters[6],MCMC_parameters[7],#Mean values of p(z|tau) multi-variate gaussian (g_ = 'given')
            MCMC_parameters[8],MCMC_parameters[9],MCMC_parameters[10],MCMC_parameters[11], 
            #
            MCMC_parameters[12:12+len(ZL_OBS)],MCMC_parameters[12+len(ZL_OBS):],# Redshifts to constrain (Note: these aren't Z_OBS)
            #
            ZL_OBS,ZS_OBS,SIGMA_ZL_OBS,SIGMA_ZS_OBS,R_OBS,SIGMA_R_OBS,SIGMA_R_OBS_2,P_TAU,COSMO_TYPE)
    print('RETURNING NOTHING')

def mcmc_SL(n,n_walkers,likelihood,zbias,mubias,OMi,Odei,H0i,wi,omstep,ode_step,H0step,wstep,db_in,fileout,status,cosmo_type,
            contaminated=None,photometric=None):
    assert contaminated is not None
    assert photometric is not None
    print(f'Assuming the sample is {"not "*(~contaminated)}contaminated, and that the redshifts are {"not "*(~photometric)}photometric')
    assert cosmo_type in ['FlatLambdaCDM','LambdaCDM','FlatwCDM','wCDM']
    if cosmo_type in ['FlatLambdaCDM','FlatwCDM']: Flat_bool=True; print('Assuming a Flat Cosmology')
    else: Flat_bool=False; print('Allowing the cosmology to have non-zero curvature')
    if cosmo_type in ['wCDM','FlatwCDM']: Complex_EoS_bool=True; print('Assuming Non-trivial DE EoS')
    else: Complex_EoS_bool=False; print('Assuming w=-1')
    ###############Data manipulation###############
    
    if zbias == 'nobias':
        columnz = 0
    elif zbias == 'bias':
        columnz = 1
    
    if mubias == 'nobias':
        columnmu = 3
    elif mubias == 'bias':
        columnmu = 4
    elif mubias == 'photometric':
        columnmu = 2
    
    zL_obs=db_in['zL_obs'].to_numpy();zS_obs=db_in['zS_obs'].to_numpy()
    #if not contaminated: r_obs=db_in['r_obs'].to_numpy()
    #if contaminated: r_obs=db_in['r_obs_contam'].to_numpy()
    '''Will always use the contaminated inputs for r_obs (they can be set to the true values if not contaminated), so can test the effect
    of accounting for the bias or not:'''
    r_obs = db_in['r_obs_contam'].to_numpy()
    sigma_r_obs=db_in['sigma_r_obs'].to_numpy() #Making these up!!!
    sigma_r_obs_2 = 1000*np.max(sigma_r_obs) #Making this a large number so contaminants are effectively ignored
    if photometric:
        sigma_zL_obs = db_in['sigma_zL_obs'].to_numpy()
        sigma_zS_obs = db_in['sigma_zS_obs'].to_numpy()
    '''
    Not sure what sigma_r_obs_0 should be if the object is not a lens?
    '''
    if contaminated: P_tau = db_in['P_tau'].to_numpy()
    rand.seed(7)

    Initialisation_dict = {'H0':{'L':50,'H':100},'OM':{'L':0,'H':1},'OD':{'L':0,'H':1},'W0':{'L':-6,'H':4},
                           'mu_zL_g_L':{'L':0,'H':0.5},'mu_zS_g_L':{'L':0.5,'H':1},
                           'si_00_g_L':{'L':0.1,'H':2},'si_11_g_L':{'L':0.1,'H':2},
                           'mu_zL_g_NL':{'L':0,'H':5},'mu_zS_g_NL':{'L':0,'H':5},
                           'si_00_g_NL':{'L':0.1,'H':5},'si_11_g_NL':{'L':0.1,'H':5},
                           'zL':{'L':0.5,'H':0.5},'zS':{'L':1,'H':1}}
    if not photometric: cur_state = np.concatenate([np.random.uniform(low=Initialisation_dict[elem]['L'],
                                                                      high=Initialisation_dict[elem]['H'],
                                size=(n_walkers,1)) for elem in ['OM','OD','H0','W0']],axis=1)
    if photometric: 
        cur_state = np.concatenate([np.random.uniform(low=Initialisation_dict[elem]['L'],
                                                      high=Initialisation_dict[elem]['H'],
                                size=(n_walkers,1)) for elem in ['OM','OD','H0','W0'] +\
                                    ['mu_zL_g_L','mu_zS_g_L','mu_zL_g_NL','mu_zS_g_NL'] +\
                                    ['si_00_g_L','si_11_g_L','si_00_g_NL','si_00_g_NL']],axis=1)
        cur_state = np.concatenate([cur_state,
                                    np.repeat([zL_obs.tolist()],n_walkers,axis=0), #Initialising at the measured redshift
                                    np.repeat([zS_obs.tolist()],n_walkers,axis=0)],axis=1) #Initialising at the measured redshift

    if contaminated:
        sigma_r_obs_2 = sigma_r_obs_2
        P_tau = P_tau
    else: 
        sigma_r_obs_2 = None
        P_tau = None

    with Pool() as pool:
        if not photometric: sampler = emcee.EnsembleSampler(n_walkers, ndim=4, log_prob_fn = likelihood_with_bounds,
                                        kwargs = {'ZL_OBS':zL_obs,'ZS_OBS':zS_obs,'R_OBS':r_obs,'SIGMA_R_OBS':sigma_r_obs,
                                                'likelihood':likelihood,
                                                'SIGMA_R_OBS_2':sigma_r_obs_2,'P_TAU':P_tau,
                                                'COSMO_TYPE':cosmo_type,'FLAT_BOOL':Flat_bool,'COMPLEX_EOS_BOOL':Complex_EoS_bool,
                                                'CONTAMINATED':contaminated,'PHOTOMETRIC':photometric},pool=pool)#,backend=backend)
        if photometric: sampler = emcee.EnsembleSampler(n_walkers, 
                                        ndim=12+2*len(r_obs),#ndim is 4 cosmo, 4 means, 4 diagonal covariances and all lens/source redshifts
                                    log_prob_fn = likelihood_with_bounds,
                                    kwargs = {
                                            'ZL_OBS':zL_obs,'ZS_OBS':zS_obs,'SIGMA_ZL_OBS':sigma_zL_obs,'SIGMA_ZS_OBS':sigma_zS_obs,
                                            'R_OBS':r_obs,'SIGMA_R_OBS':sigma_r_obs,
                                            'likelihood':likelihood,
                                            'SIGMA_R_OBS_2':sigma_r_obs_2,'P_TAU':P_tau,
                                            'COSMO_TYPE':cosmo_type,'FLAT_BOOL':Flat_bool,'COMPLEX_EOS_BOOL':Complex_EoS_bool,
                                            'CONTAMINATED':contaminated,'PHOTOMETRIC':photometric},pool=pool)#,backend=backend)
        _ = sampler.run_mcmc(cur_state,n,progress=True,skip_initial_state_check=True)
    save_time = time.time()
    if not photometric: np.save(f'{fileout}_mcmc_chains_{save_time}.npy',sampler.chain)
    else: np.save(f'{fileout}_mcmc_chains_{save_time}.npy',sampler.chain[:,:,0:-2*int(len(r_obs))]) #Not saving the redshift values to save space
    return 
    ###############MCMC code (bias)###############
    
    omlist = [] #create empty list
    H0list = []
    odelist = []
    wlist = []
    if Flat_bool: 
        assert OMi+Odei==1 #Assert cosmology is flat
    if not Complex_EoS_bool: 
        assert wi==-1 #Assert EoS of DE is -1.

    om_current = OMi #starting values
    ode_current = Odei
    H0_current = H0i
    w_current = wi
    
    omlist.append(om_current) #append first value to list
    H0list.append(H0_current)
    odelist.append(ode_current)
    wlist.append(w_current)
    sig_tau = 0.2 #Where this from???
    accept = 0
    log_like_proposed = 0
    accept_list = []
    
    print('Generating posterior')
    
    fout = open(fileout,'w')
    fout.write('#o_m\tH0\tw\tOde\n')
    for i in range(0,n-1):
        #current position:
        if i == 0: 
            if not contaminated: log_like_current = likelihood(om_current,ode_current,H0_current,w_current,zL_obs,zS_obs,r_obs,sigma_r_obs,cosmo_type)
            if contaminated: log_like_current =  likelihood(om_current,ode_current,H0_current,w_current,zL_obs,zS_obs,r_obs,sigma_r_obs,sigma_r_obs_2,P_tau,cosmo_type)
        elif accept == 1:
            log_like_current = log_like_proposed
        
        #OM
        om_proposed =  rand.gauss(om_current,omstep)
        while om_proposed >= 1 or om_proposed <= 0:         #keeps Omega_matter in (0,1) 
            om_proposed =  rand.gauss(om_current,omstep)    #for numerical reasons
        #Ode
        if Flat_bool:
            ode_proposed = 1-om_proposed;
        else: #If not flat, allow Ode to vary too:
            assert cosmo_type in ['wCDM','LambdaCDM']
            ode_proposed = rand.gauss(ode_current,ode_step)
            while ode_proposed>=1 or ode_proposed<=0: #keeps Omega_de in (0,1)
                ode_proposed = rand.gauss(ode_current,ode_step)
        #H0
        H0_proposed =  rand.gauss(H0_current,H0step)
        while H0_proposed >= 200 or H0_proposed <= 10:
            H0_proposed =  rand.gauss(H0_current,H0step)
        #w0
        if Complex_EoS_bool:
            w_proposed =  rand.gauss(w_current,wstep)
            while w_proposed >= 4 or w_proposed <= -6:
                w_proposed =  rand.gauss(w_current,wstep)
        else:
            assert cosmo_type in ['FlatLambdaCDM','LambdaCDM']
            w_proposed=-1
            
        #proposed position:
        if not contaminated: log_like_proposed = likelihood(om_proposed,ode_proposed,H0_proposed,w_proposed,zL_obs,zS_obs,r_obs,sigma_r_obs,cosmo_type)
        if contaminated: log_like_proposed = likelihood(om_proposed,ode_proposed,H0_proposed,w_proposed,zL_obs,zS_obs,r_obs,sigma_r_obs,sigma_r_obs_2,P_tau,cosmo_type)
        #decision:
        r = np.exp(log_like_proposed - log_like_current)
        
        MC = rand.random()
        
        if r < 1 and MC > r:
            omlist.append(om_current)
            H0list.append(H0_current)
            odelist.append(ode_current)
            wlist.append(w_current)
            accept = 0
        else:
            omlist.append(om_proposed)
            H0list.append(H0_proposed)
            odelist.append(ode_proposed)
            wlist.append(w_proposed)
            accept = 1
            
        om_current = omlist[i+1]
        H0_current = H0list[i+1]
        ode_current = odelist[i+1]
        w_current = wlist[i+1]
        accept_list.append(accept)
        
        fout.write(str(om_current)+'\t'+str(H0_current)+'\t'+str(w_current)+'\t'+str(ode_current)+'\n')
        
        if status==True:
            inc100 = np.int64(i/(n-2)*100)
            inc50 = np.int64(i/(n-2)*50)
            sys.stdout.write('\r')
            sys.stdout.write('[' + '#'*inc50 + ' '*(50-inc50) + ']' + str(inc100) + '%')
            sys.stdout.flush()
            
    fout.close()
    print('\ndone')


def mcmc_spec(n,likelihood,mubias,OMi,H0i,wi,omstep,H0step,wstep,filein,fileout,status):
    
    ###############Data manipulation###############
    
    if mubias == 'nobias':
        columnmu = 3
    elif mubias == 'bias':
        columnmu = 4
    
    z_obs,z_obs2,mu_obs = np.loadtxt(filein,usecols=[0,1,columnmu],unpack=True)
    
    rand.seed(7)
    
    
    ###############MCMC code (bias)###############
    
    omlist = [] #create empty list
    H0list = []
    wlist = []
    
    om_current = OMi #starting values
    H0_current = H0i
    w_current = wi
    
    omlist.append(om_current) #append first value to list
    H0list.append(H0_current)
    wlist.append(w_current)
    
    P_gamma = 0.91
    P_tau = 0.95
    offset = 2
    sig_tau1 = 0.2
    sig_tau2 = 1.5
    
    accept = 0
    log_like_proposed = 0
    accept_list = []
    
    print('Generating posterior')
    
    fout = open(fileout,'w')
    fout.write('#o_m \t H0 \t w \n')
    for i in range(0,n-1):
        #current position:
        if i == 0:
            log_like_current = likelihood(z_obs,z_obs2,mu_obs,sig_tau1,sig_tau2,
                                          P_gamma,P_tau,offset,om_current,H0_current,w_current)
        elif accept == 1:
            log_like_current = log_like_proposed
            
        om_proposed =  rand.gauss(om_current,omstep)
        while om_proposed >= 1 or om_proposed <= 0:         #keeps Omega_matter in (0,1) 
            om_proposed =  rand.gauss(om_current,omstep)    #for numerical reasons
        H0_proposed =  rand.gauss(H0_current,H0step)
        while H0_proposed >= 200 or H0_proposed <= 10:
            H0_proposed =  rand.gauss(H0_current,H0step)
        w_proposed =  rand.gauss(w_current,wstep)
        while w_proposed >= 4 or w_proposed <= -6:
            w_proposed =  rand.gauss(w_current,wstep)
            
        #proposed position:
        log_like_proposed = likelihood(z_obs,z_obs2,mu_obs,sig_tau1,sig_tau2,
                                       P_gamma,P_tau,offset,om_proposed,H0_proposed,w_proposed)
        
        #decision:
        r = np.exp(log_like_proposed - log_like_current)
        
        MC = rand.random()
        
        if r < 1 and MC > r:
            omlist.append(om_current)
            H0list.append(H0_current)
            wlist.append(w_current)
            accept = 0
        else:
            omlist.append(om_proposed)
            H0list.append(H0_proposed)
            wlist.append(w_proposed)
            accept = 1
            
        om_current = omlist[i+1]
        H0_current = H0list[i+1]
        w_current = wlist[i+1]
        accept_list.append(accept)
        
        fout.write(str(om_current)+'\t'+str(H0_current)+'\t'+str(w_current)+'\n')
        
        if status==True:
            inc100 = np.int64(i/(n-2)*100)
            inc50 = np.int64(i/(n-2)*50)
            sys.stdout.write('\r')
            sys.stdout.write('[' + '#'*inc50 + ' '*(50-inc50) + ']' + str(inc100) + '%')
            sys.stdout.flush()
            
    fout.close()
    print('\ndone')

def mcmc_phot(n,block,burnin,thinning,likelihood,OMi,H0i,wi,bi,
              omstep_i,H0step_i,wstep_i,zstep_i,bstep,filein,fileout,status):
    
    ###############MCMC code###############
    '''From zBEAMS paper: We used a block Metropolis-Hastings sampling method — aﬀectionately dubbed “Arabian nights” to ﬁt for 1001 parameters
    simultaneously (3 cosmological parameters and 998 redshifts) The block Metropolis-Hastings proceeds identically to the usual 
    Metropolis-Hastings sampling algorithm, except that parameters are updated in blocks instead of updating all parameters every step.'''
    z,mu,sig_mu = np.loadtxt(filein,usecols=[1,2,3],unpack=True)
    fout = open(fileout,'w')
    fout.write('#OM \t\t\t H0 \t\t\t w \n')
    
    dz = 0.04
    '''From zBEAMS paper: We assume for this work that the redshift uncertainties are Gaussian distributed with a standard deviation
    of 0.04(1 + z), though any more realistic distribution can be assumed with little change in complexity.    
    '''
    zdisp = dz*(1+z.copy())
    block = int(block)
    
    OMlist = [] #create empty list
    H0list = []
    wlist = []
    accept_list = []
    
    OM_current = OMi #starting values
    H0_current = H0i
    w_current = wi
    z_current = z.copy()
    b_current = bi
    
    OMlist.append(OM_current) #append first value to list
    H0list.append(H0_current)
    wlist.append(w_current)
    
    print('Generating posterior')
    accept = 0
    log_like_proposed = 0
    lenz = len(z)
    rand.seed(7)
    for i in range(0,n-1):
        #current position:
        if i <= burnin:
            OMstep = 0 #step sizes
            H0step = 0
            wstep = 0
            zstep = 2
        else:
            OMstep = omstep_i #step sizes
            H0step = H0step_i
            wstep = wstep_i
            zstep = zstep_i
            
        if i == 0:
            log_like_current = likelihood(z_current,mu,sig_mu,OM_current,H0_current,w_current,z,zdisp,b_current)
        elif accept == 1:
            log_like_current = log_like_proposed
            
        OM_proposed =  rand.gauss(OM_current,OMstep)
        while OM_proposed >= 1 or OM_proposed <= 0:         #keeps Omega_matter in (0,1) 
            OM_proposed =  rand.gauss(OM_current,OMstep)    #for numerical reasons
        H0_proposed =  rand.gauss(H0_current,H0step)
        while H0_proposed >= 200 or H0_proposed <= 10:
            H0_proposed =  rand.gauss(H0_current,H0step)
        w_proposed =  rand.gauss(w_current,wstep)
        while w_proposed >= 4 or w_proposed <= -6:
            w_proposed =  rand.gauss(w_current,wstep)
        #Useful MCMC video (on Metropolis-Hastings Algorithm) here: https://youtu.be/yCv2N7wGDCw?si=IrWmM0jp3bfQmE-6
        MC1 = [rand.randint(0,lenz-1) for j in range(block)]
        z_proposed = z_current.copy() #Length equal to number of datapoints
        z_proposed[MC1] = [rand.gauss(z_current[MC1],(zstep*zdisp[MC1])**2) for j in range(block)] #Only change one of the redshifts at a time?
        b_proposed = rand.gauss(b_current,bstep)
        #proposed position:
        log_like_proposed=likelihood(z_proposed,mu,sig_mu,OM_proposed,H0_proposed,w_proposed,z,zdisp,b_proposed)
        
        #decision:
        r = np.exp(log_like_proposed - log_like_current) #Is the proposed position more or less likely than the current one? This is 
        #r_f in the video tutorial above.
        MC0 = rand.random()
        
        if r < 1 and MC0 > r: #If less likely, AND less likely than a given random number, reject it: rand<r<1 => reject. In practice, the ratio
            # r is the ratio of p(proposed)/p(current) (i.e. the ratio of likelihoods). E.g if r 0.9 it will still be accepted 90% of
            # the time (as r will be > random_number 90% of the time). Increasingly unlikely proposals are increasingly unlikely to be accepted. 
            OMlist.append(OM_current)
            H0list.append(H0_current)
            wlist.append(w_current)
            accept = 0
            like = log_like_current
        else: #Otherwise accept the proposal:
            OMlist.append(OM_proposed)
            H0list.append(H0_proposed)
            wlist.append(w_proposed)
            z_current = z_proposed.copy()
            b_current = b_proposed
            accept = 1
            like = log_like_proposed
            
        accept_list.append(accept)
        OM_current = OMlist[i+1]
        H0_current = H0list[i+1]
        w_current = wlist[i+1]
        
        if i >= burnin and i%thinning == 0:
            fout.write(f'{OM_current}\t{H0_current}\t{w_current}\t{like}')
            fout.write('\n')
    
        if status==True:
            inc100 = np.int64(i/(n-2)*100)
            inc50 = np.int64(i/(n-2)*50)
            sys.stdout.write('\r')
            sys.stdout.write('[' + '#'*inc50 + ' '*(50-inc50) + ']' + str(inc100) + '%')
            sys.stdout.flush()
            
    fout.close()
    print('\ndone')