import sys
#sys.path.append('/Users/hollowayp/zBEAMS')
sys.path.append('/mnt/zfsusers/hollowayp/zBEAMS')

from astropy.cosmology import LambdaCDM,FlatLambdaCDM,wCDM,FlatwCDM,w0waCDM
from Lenstronomy_Cosmology import Background, LensCosmo
from jax_cosmo import Cosmology
from jax_cosmo import background_flat as background
import matplotlib.pyplot as pl
import jax.numpy as jnp
import jax_cosmo as jc
import numpy as np
import jax

'''
Have renamed some functions using astropy (e.g. r_SL) to ..._astr to prevent them overwriting any functions with these names outside this file.
'''

def j_D_cov(z,j_cosmo):
    #Comoving distance in Mpc
    return background.radial_comoving_distance(j_cosmo,jc.utils.z2a(z))/j_cosmo.h #**Dividing** by h as otherwise returned in Mpc/h.

def j_D_cov_z1z2(zL,zS,j_cosmo):
    return j_D_cov(zS,j_cosmo)-j_D_cov(zL,j_cosmo)

def j_D_S(zS,j_cosmo):
    return background.angular_diameter_distance(j_cosmo,jc.utils.z2a(zS))/j_cosmo.h#.T  #**Dividing** by h as otherwise returned in Mpc/h.

def D_S_astr(zL,zS,cosmo):
    LensCosmo_i = LensCosmo(z_lens=zL,z_source=zS,cosmo=cosmo)
    D_S = LensCosmo_i.ds
    return D_S

#jnp.where(some_factor,val_if_true,val_if_false)
def j_D_LS(zL,zS,j_cosmo):
    D_cov_ZL_ZS = j_D_cov_z1z2(zL,zS,j_cosmo)
    dH = jc.constants.c/(100*j_cosmo.h) #(km/s)*(km/(s*Mpc))^-1 = (1/s)*s*Mpc = Mpc.
    sqrt_Ok = jnp.sqrt(abs(j_cosmo.Omega_k))
    if True:#j_cosmo.Omega_k<0:
        a1 = (1/(1+zS))#[...,jnp.newaxis]
        b1 = jnp.sin(sqrt_Ok*D_cov_ZL_ZS/dH)#jnp.squeeze(jnp.sin(sqrt_Ok*D_cov_ZL_ZS/dH),2)
        c1 = (dH/sqrt_Ok)#.T
#        return a1*b1*c1
    if True:#j_cosmo.Omega_k==0:
        a2 = (1/(1+zS))#[...,jnp.newaxis]
        b2 = jnp.squeeze(D_cov_ZL_ZS)
#        return a2*b2
    if True:#j_cosmo.Omega_k>0:
        a3 = (1/(1+zS))#[...,jnp.newaxis]
        b3 = jnp.sinh(sqrt_Ok*D_cov_ZL_ZS/dH)#jnp.squeeze(jnp.sinh(sqrt_Ok*D_cov_ZL_ZS/dH),2)
        c3 = (dH/sqrt_Ok)#.T
#        return a3*b3*c3
    #Note: k takes opposite sign to Omega_k
    j_D_LS_ii = a2*b2
    return j_D_LS_ii
#    return (a1*b1*c1)*(j_cosmo.Omega_k<0).T+(a2*b2)*(j_cosmo.Omega_k==0).T+(a3*b3*c3)*(j_cosmo.Omega_k>0).T

def D_LS_astr(zL,zS,cosmo):
    LensCosmo_i = LensCosmo(z_lens=zL,z_source=zS,cosmo=cosmo)
    D_LS = LensCosmo_i.dds
    return D_LS

def j_r_SL(zL,zS,j_cosmo): 
    D_LS = j_D_LS(zL,zS,j_cosmo)
    D_S = j_D_S(zS,j_cosmo)
    return D_LS/D_S

def r_SL_astr(zL,zS,cosmo):
    LensCosmo_i = LensCosmo(z_lens=zL,z_source=zS,cosmo=cosmo)
    D_LS = LensCosmo_i.dds
    D_S = LensCosmo_i.ds
    r_value = D_LS.value/D_S.value
    if np.isnan(r_value).any():
        print('Redshift input',zL,zS,zL<zS)
        assert False
    return r_value

def D_cov_check(z,cosmo,j_cosmo,plot=False):
    D_cov_i = cosmo.comoving_distance(z).to_value()
    j_D_cov_i = j_D_cov(z,j_cosmo)#[:,indx,0]
    #print('numpy',D_cov_i[0:100:10])
    #print('jax',j_D_cov_i[0:100:10])
    #print('cosmo',cosmo)
    #print('JD_Cov',j_D_cov_i)
    if plot: 
        pl.plot(z,D_cov_i)
        pl.plot(z,j_D_cov_i,'--')
        pl.show()
    assert ((abs(D_cov_i-j_D_cov_i)/D_cov_i)<0.01).all()

def D_LS_check(zL,zS,cosmo,j_cosmo,plot=False):
    D_LS_i = cosmo.angular_diameter_distance_z1z2(zL,zS).to_value()
    #print('All lenses lower redshift',(zL<=zS).all())
    j_D_LS_i = j_D_LS(zL,zS,j_cosmo)#[:,indx]
    if plot:
        fig,ax = pl.subplots(1,2,figsize=(10,5))
        ax[0].plot(zL,D_LS_i,label='Astropy')
        ax[0].plot(zL,j_D_LS_i,'--',label='JAX')
        ax[0].legend();ax[0].set_ylabel('D_LS')
        ax[1].plot(zL,D_LS_i-j_D_LS_i)
        ax[1].set_ylabel('D_LS: Astropy-JAX')
        pl.show()
    try:
        assert ((abs(D_LS_i-j_D_LS_i)/D_LS_i)<0.01).all()  
    except Exception as ex:
        print('Failed Check:',ex)
        print('Cosmology:',cosmo)
        fig,ax = pl.subplots(1,2,figsize=(10,5))
        ax[0].plot(zL,D_LS_i,label='Astropy')
        ax[0].plot(zL,j_D_LS_i,'--',label='JAX')
        ax[0].legend();ax[0].set_ylabel('D_LS')
        ax[1].plot(zL,D_LS_i-j_D_LS_i)
        ax[1].set_ylabel('D_LS: Astropy-JAX')
        pl.suptitle('Failed: D_LS_check')
        pl.show()

def r_SL_check(zL,zS,cosmo,j_cosmo,plot=False):
    r_SL_i = r_SL_astr(zL,zS,cosmo)
    j_r_SL_i = j_r_SL(zL,zS,j_cosmo)#[:,indx]
    if plot:
        fig,ax = pl.subplots(1,2,figsize=(10,5))
        ax[0].plot(zL,r_SL_i,label='Astropy')
        ax[0].plot(zL,j_r_SL_i,'--',label='JAX')
        ax[0].legend();ax[0].set_ylabel('r_SL')
        ax[1].plot(zL,r_SL_i-j_r_SL_i)
        ax[1].ylabel('r_SL: Astropy-JAX')
        pl.show()
    try:
        assert ((abs(r_SL_i-j_r_SL_i)/r_SL_i)<0.01).all()
    except Exception as ex:
        print('Failed Check (2)',ex)
        print('Cosmology:',cosmo)
        fig,ax = pl.subplots(1,2,figsize=(10,5))
        ax[0].plot(zL,r_SL_i,label='Astropy')
        ax[0].plot(zL,j_r_SL_i,'--',label='JAX')
        ax[0].legend();ax[0].set_ylabel('r_SL')
        ax[1].plot(zL,r_SL_i-j_r_SL_i)
        ax[1].set_ylabel('r_SL: Astropy-JAX')
        pl.suptitle('Failed: r_SL_check')
        pl.show()
    
def cosmo_check():
    print('Running Cosmo Check')
    print("NOTE: This cosmology still needs fixing - doesn't work for large values of Ode, but need to check if these are physical values anyway")
    H0_sim = jnp.array([50,60,70])[...,jnp.newaxis]
    Om_sim = jnp.array([0.2,0.3,0.4])[...,jnp.newaxis]
    Ok_sim = jnp.array([0,0.0,0])[...,jnp.newaxis]
    Ode_sim = 1-(Om_sim+Ok_sim) #NOTE: Ode can't be negative (O_k can be), so don't be surprised if it bugs if it is!
    assert jnp.shape(H0_sim)==jnp.shape(Ode_sim)
    #
    H0_com = jnp.array([50,60,70])[...,jnp.newaxis]
    Om_com = jnp.array([0.2,0.3,0.4])[...,jnp.newaxis]
    Ok_com = jnp.array([0.0,0.0,0.0])[...,jnp.newaxis]
    Ode_com = 1-(Om_com+Ok_com) #NOTE: Ode can't be negative (O_k can be), so don't be surprised if it bugs if it is!
    w0_com = jnp.array([-0.5,-0.5,1.5])[...,jnp.newaxis]
    #
    H0_vcom = jnp.array([50,60,70])[...,jnp.newaxis]
    Om_vcom = jnp.array([0.2,0.3,0.1])[...,jnp.newaxis]
    Ok_vcom = jnp.array([0.0,0.0,0.0])[...,jnp.newaxis]
    Ode_vcom = 1-(Om_vcom+Ok_vcom) #NOTE: Ode can't be negative (O_k can be), so don't be surprised if it bugs if it is!
    w0_vcom = jnp.array([-0.5,-0.5,1.5])[...,jnp.newaxis]
    wa_vcom = jnp.array([-0.9,0.1,0.9])[...,jnp.newaxis]
    #
    zL_check = np.linspace(0.05,2,101)
    zS_check = np.linspace(0.3,5,101)
    for complexity in range(3):
        for cosmo_iter in range(len([H0_sim,H0_com,H0_vcom][complexity])):
            print('complexity',complexity,'cosmo_iter',cosmo_iter)
            cosmo_simple = LambdaCDM(
                                H0=H0_sim[cosmo_iter].tolist()[0], 
                                Om0=Om_sim[cosmo_iter].tolist()[0],
                                Ode0=Ode_sim[cosmo_iter].tolist()[0])
            j_cosmo_simple = jc.Cosmology(
                                Omega_c=Om_sim[cosmo_iter].tolist()[0],
                                h=H0_sim[cosmo_iter].tolist()[0]/100,
                                Omega_k=Ok_sim[cosmo_iter].tolist()[0],
                                Omega_b=0.0, w0=-1., wa=0.,sigma8 = 0.8, n_s=0.96)
            cosmo_complex = wCDM(
                                H0=H0_com[cosmo_iter].tolist()[0], 
                                Om0=Om_com[cosmo_iter].tolist()[0],
                                Ode0=Ode_com[cosmo_iter].tolist()[0], 
                                w0=w0_com[cosmo_iter].tolist()[0])
            j_cosmo_complex = jc.Cosmology(
                                Omega_c=Om_com[cosmo_iter].tolist()[0],
                                h=H0_com[cosmo_iter].tolist()[0]/100,
                                Omega_k=Ok_com[cosmo_iter].tolist()[0],
                                w0=w0_com[cosmo_iter].tolist()[0], wa=0.,
                                Omega_b=0.0, sigma8 = 0.8, n_s=0.96)
            cosmo_very_complex = w0waCDM(
                                H0=H0_vcom[cosmo_iter].tolist()[0],
                                Om0=Om_vcom[cosmo_iter].tolist()[0],
                                Ode0=Ode_vcom[cosmo_iter].tolist()[0], 
                                w0=w0_vcom[cosmo_iter].tolist()[0],
                                wa=wa_vcom[cosmo_iter].tolist()[0])
            j_cosmo_very_complex = jc.Cosmology(
                                    Omega_c=Om_vcom[cosmo_iter].tolist()[0],
                                    h=H0_com[cosmo_iter].tolist()[0]/100,
                                    Omega_k=Ok_vcom[cosmo_iter].tolist()[0],
                                    w0=w0_vcom[cosmo_iter].tolist()[0],
                                    wa=wa_vcom[cosmo_iter].tolist()[0],
                                    Omega_b=0.0,sigma8 = 0.8, n_s=0.96)
            cosmo_i = [cosmo_simple,cosmo_complex,cosmo_very_complex][complexity]
            j_cosmo_i = [j_cosmo_simple,j_cosmo_complex,j_cosmo_very_complex][complexity]
            for z_i in [zL_check,zS_check]:
                #print('redshifts',z_i)
                D_cov_check(z_i,cosmo_i,j_cosmo_i,plot=False)
            D_LS_check(zL_check,zS_check,cosmo_i,j_cosmo_i,plot=False)
            r_SL_check(zL_check,zS_check,cosmo_i,j_cosmo_i,plot=False)

#Do ***NOT*** remove this line: it is vital to check the jax cosmology is implemented correctly!
cosmo_check()
#Seems to break for values of k close to 1?