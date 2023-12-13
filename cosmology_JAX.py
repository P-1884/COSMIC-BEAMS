import sys
#sys.path.append('/Users/hollowayp/zBEAMS')
sys.path.append('/mnt/zfsusers/hollowayp/zBEAMS')
from Lenstronomy_Cosmology import Background, LensCosmo

from jax_cosmo import Cosmology, background
import jax.numpy as jnp
import jax_cosmo as jc
import jax

from astropy.cosmology import LambdaCDM,FlatLambdaCDM,wCDM,FlatwCDM,w0waCDM
import matplotlib.pyplot as pl
import numpy as np

'''
Have renamed some functions using astropy (e.g. r_SL) to ..._astr to prevent them overwriting any functions with these names outside this file.
'''

def j_D_cov(z,j_cosmo):
    #Comoving distance in Mpc
    return background.radial_comoving_distance(j_cosmo,jc.utils.z2a(z))/j_cosmo.h #**Dividing** by h as otherwise returned in Mpc/h.

def j_D_cov_z1z2(zL,zS,j_cosmo):
    return j_D_cov(zS,j_cosmo)-j_D_cov(zL,j_cosmo)

def j_D_S(zS,j_cosmo):
    return background.angular_diameter_distance(j_cosmo,jc.utils.z2a(zS))/j_cosmo.h  #**Dividing** by h as otherwise returned in Mpc/h.

def D_S_astr(zL,zS,cosmo):
    LensCosmo_i = LensCosmo(z_lens=zL,z_source=zS,cosmo=cosmo)
    D_S = LensCosmo_i.ds
    return D_S

#jnp.where(some_factor,val_if_true,val_if_false)
def j_D_LS(zL,zS,j_cosmo):
    D_cov_ZL_ZS = j_D_cov_z1z2(zL,zS,j_cosmo)
    dH = jc.constants.c/(100*j_cosmo.h) #(km/s)*(km/(s*Mpc))^-1 = (1/s)*s*Mpc = Mpc.
    sqrt_Ok = jnp.sqrt(abs(j_cosmo.Omega_k))
    if j_cosmo.Omega_k<0:
        return (1/(1+zS))*jnp.sin(sqrt_Ok*D_cov_ZL_ZS/dH)*dH/sqrt_Ok
    if j_cosmo.Omega_k==0:
        return (1/(1+zS))*D_cov_ZL_ZS
    if j_cosmo.Omega_k>0:
        return (1/(1+zS))*jnp.sinh(sqrt_Ok*D_cov_ZL_ZS/dH)*dH/sqrt_Ok

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
    j_D_cov_i = j_D_cov(z,j_cosmo)
    if plot: 
        pl.plot(z,D_cov_i)
        pl.plot(z,j_D_cov_i,'--')
        pl.show()
    print('SHAPE',np.shape(D_cov_i),np.shape(j_D_cov_i))
    print((abs(D_cov_i-j_D_cov_i)/D_cov_i))
    assert ((abs(D_cov_i-j_D_cov_i)/D_cov_i)<0.01).all()

def D_LS_check(zL,zS,cosmo,j_cosmo,plot=False):
    D_LS_i = cosmo.angular_diameter_distance_z1z2(zL,zS).to_value()
    j_D_LS_i = j_D_LS(zL,zS,j_cosmo)
    if plot:
        pl.plot(zL,D_LS_i)
        pl.plot(zL,j_D_LS_i,'--')
        pl.show()
    assert ((abs(D_LS_i-j_D_LS_i)/D_LS_i)<0.01).all()    

def r_SL_check(zL,zS,cosmo,j_cosmo,plot=False):
    r_SL_i = r_SL_astr(zL,zS,cosmo)
    j_r_SL_i = j_r_SL(zL,zS,j_cosmo)
    if plot:
        pl.plot(zL,r_SL_i)
        pl.plot(zL,j_r_SL_i,'--')
        pl.show()
    assert ((abs(r_SL_i-j_r_SL_i)/r_SL_i)<0.01).all()    

def cosmo_check():
    cosmo_simple = LambdaCDM(H0=60, Om0=0.3, Ode0=0.6) #Ok = 0.1
    j_cosmo_simple = jc.Cosmology(Omega_c=0.3,Omega_b=0.0, h=0.6, Omega_k=0.1, w0=-1., wa=0.,sigma8 = 0.8, n_s=0.96)
    #
    cosmo_complex = wCDM(H0=50, Om0=0.3, Ode0=0.3, w0=-1.5) #Ok = 0.4
    j_cosmo_complex = jc.Cosmology(Omega_c=0.3,Omega_b=0.0,h=0.5,Omega_k=0.4, w0=-1.5, wa=0.,sigma8 = 0.8, n_s=0.96)
    #
    cosmo_very_complex = w0waCDM(H0=40, Om0=0.2, Ode0=0.3, w0=-1.5, wa=0.9)
    j_cosmo_very_complex = jc.Cosmology(Omega_c=0.2,Omega_b=0.0,h=0.4,Omega_k=0.5, w0=-1.5, wa=0.9,sigma8 = 0.8, n_s=0.96)
    zL_check = np.linspace(0.05,2,101)
    zS_check = np.linspace(0.3,5,101)
    for c_i in range(3):
        cosmo_i = [cosmo_simple,cosmo_complex,cosmo_very_complex][c_i]
        j_cosmo_i = [j_cosmo_simple,j_cosmo_complex,j_cosmo_very_complex][c_i]
        for z_i in [zL_check,zS_check]:
            D_cov_check(z_i,cosmo_i,j_cosmo_i,plot=False)
        D_LS_check(zL_check,zS_check,cosmo_i,j_cosmo_i,plot=False)
        r_SL_check(zL_check,zS_check,cosmo_i,j_cosmo_i,plot=False)

#Do ***NOT*** remove this line: it is vital to check the jax cosmology is implemented correctly!
cosmo_check()