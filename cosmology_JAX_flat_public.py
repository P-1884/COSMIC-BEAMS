from astropy.cosmology import LambdaCDM,FlatLambdaCDM,wCDM,FlatwCDM,w0waCDM
from Lenstronomy_Cosmology import Background, LensCosmo
from jax_cosmo import Cosmology
from jax_cosmo import background_flat as background
import matplotlib.pyplot as pl
import jax.numpy as jnp
import jax_cosmo as jc
import numpy as np
import jax
import sys
from jax import jit
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

def j_D_LS(zL,zS,j_cosmo):
    D_cov_ZL_ZS = j_D_cov_z1z2(zL,zS,j_cosmo)
    dH = jc.constants.c/(100*j_cosmo.h) #(km/s)*(km/(s*Mpc))^-1 = (1/s)*s*Mpc = Mpc.
    sqrt_Ok = jnp.sqrt(abs(j_cosmo.Omega_k))
    a1 = (1/(1+zS))
    b1 = jnp.sin(sqrt_Ok*D_cov_ZL_ZS/dH)
    c1 = (dH/sqrt_Ok)
    ###
    a2 = (1/(1+zS))
    b2 = jnp.squeeze(D_cov_ZL_ZS)
    ###
    a3 = (1/(1+zS))
    b3 = jnp.sinh(sqrt_Ok*D_cov_ZL_ZS/dH)
    c3 = (dH/sqrt_Ok)
    #Note: k takes opposite sign to Omega_k
    j_D_LS_ii = a2*b2
    return j_D_LS_ii

def j_r_SL(zL,zS,j_cosmo): 
    D_LS = j_D_LS(zL,zS,j_cosmo)
    D_S = j_D_S(zS,j_cosmo)
    return D_LS/D_S
