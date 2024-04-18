import pandas as pd
import numpy as np
from zbeamsfunctions_SL import r_SL
from astropy.cosmology import LambdaCDM,FlatLambdaCDM,wCDM,FlatwCDM,w0waCDM
from scipy.stats import truncnorm, norm

def generate_catalogue_databases(db,N_obs,true_lens_frac,cosmo_type,Om_fid,Ode_fid,H0_fid,w_fid,wa_fid,
                                 verbose=False,Percentage_error=np.nan,
                                 absolute_error = np.nan,db_name = ''):
    assert np.isnan(Percentage_error) #Don't use this - it causes errors in the MCMC cosmology inference.
    np.random.seed(1)
    print(f'Generating sample of {N_obs} systems, of which {np.round(100*true_lens_frac,2)}% are true lenses '+\
              f'with {absolute_error} absolute measurement error in a {cosmo_type} cosmology. This has '+\
              f'parameters Om:{Om_fid}, Ode:{Ode_fid}, H0:{H0_fid}, w:{w_fid} and wa:{wa_fid}')
    #
    zL_obs_true = db['zL'].to_numpy()
    zS_obs_true = db['zS'].to_numpy()
    #I could use r_obs calculated from the database, but this wouldn't allow me to change the cosmology
    #to infer. Instead I am recalculating it. It makes minor differences to the observed value of r_obs
    #but these may be significant when infering the unbiased cosmology.
    if cosmo_type == 'w0CDM': r_obs_true = r_SL(zL_obs_true,zS_obs_true,wCDM(H0=H0_fid,Om0=Om_fid,Ode0=Ode_fid,w0=w_fid))
    if cosmo_type == 'wCDM': r_obs_true = r_SL(zL_obs_true,zS_obs_true,w0waCDM(H0=H0_fid,Om0=Om_fid,Ode0=Ode_fid,w0=w_fid,wa=wa_fid))
    if cosmo_type == 'FlatwCDM': r_obs_true = r_SL(zL_obs_true,zS_obs_true,FlatwCDM(H0=H0_fid,Om0=Om_fid,w0=w_fid))
    if cosmo_type == 'LambdaCDM': r_obs_true = r_SL(zL_obs_true,zS_obs_true,LambdaCDM(H0=H0_fid,Om0=Om_fid,Ode0=Ode_fid))
    if cosmo_type == 'FlatLambdaCDM': r_obs_true = r_SL(zL_obs_true,zS_obs_true,FlatLambdaCDM(H0=H0_fid,Om0=Om_fid))
    #Not using r_obs inferred from the catalogue:
    #r_obs_true = r_obs_func(db['tE'],db['sig_v']).to_numpy()
    #
    #Breaking the measurements of r_obs if the object is contaminated:
    sigma_r_obs_0 = absolute_error#abs(Percentage_error/100)*r_obs_true
    sigma_zL_obs = absolute_error#abs(Percentage_error/100)*zL_obs_true
    sigma_zS_obs = absolute_error#abs(Percentage_error/100)*zS_obs_true
    p_tau_0 = [np.random.choice([true_lens_frac]) for i in range(len(r_obs_true))]
    contaminated_bool_0 = [np.random.choice([0,1],p=[p_tau_0[i],1-p_tau_0[i]]) for i in range(len(r_obs_true))] #Lens = 0
    #
    r_obs_0 = norm(loc=r_obs_true,scale=sigma_r_obs_0).rvs() #Draw observations randomly from gaussian 
    r_obs_0_contam = [r_obs_0[i] if contaminated_bool_0[i] == 0 else np.random.random() for i in range(len(r_obs_true))]
    #
    zL_obs_0 = truncnorm(loc = zL_obs_true,scale = sigma_zL_obs,a=-zL_obs_true/sigma_zL_obs,b = np.inf).rvs()
    zS_obs_0 = truncnorm(loc = zS_obs_true,scale = sigma_zS_obs,a=-zS_obs_true/sigma_zS_obs,b = np.inf).rvs()
    '''
    Not sure what sigma_r_obs_0 should be if the object is not a lens?
    '''
    db_in = pd.DataFrame({'zL_obs':zL_obs_0,'zS_obs':zS_obs_0,
                        'zL_true':zL_obs_true,'zS_true':zS_obs_true,
                        'sigma_zL_obs':sigma_zL_obs,'sigma_zS_obs':sigma_zS_obs,
                        #'r_obs_if_true':r_obs_0, Not using this as its confusing if the system isn't actually a lens.
                        'r_obs_contam': r_obs_0_contam,
                        'sigma_r_obs':sigma_r_obs_0,'P_tau':p_tau_0,'FP_bool':contaminated_bool_0})
    random_indx = np.random.choice(np.arange(len(db)),replace=False,size=N_obs)
    db_in = db_in.loc[random_indx] #Randomly select the correct number of systems
#    db_filename = f'./databases/spectroscopic_db_{Percentage_error}perc_{N_obs}_samples_'+\
    absolute_error = str(absolute_error).replace('.','p')
    db_filename = f'./databases/{db_name}_Gaussian_noise_{absolute_error}abs_{N_obs}_samples_'+\
                  f'{int(np.round(100*db_in["P_tau"].mean()))}_true_cosmo_{cosmo_type}.csv' 
    print(db_filename)
    db_in.to_csv(db_filename)
    if verbose: print(db_in.describe())