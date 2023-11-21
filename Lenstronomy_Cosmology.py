#From Lenstronomy: https://github.com/lenstronomy/lenstronomy/blob/fd3f872a8802046556d595634256b590a8d58364/lenstronomy/Cosmo/background.py

from lenstronomy.Cosmo.cosmo_interp import CosmoInterp
from lenstronomy.Cosmo.background import Background
from lenstronomy.Cosmo.nfw_param import NFWParam
import lenstronomy.Util.constants as const
import matplotlib.pyplot as pl
from scipy.stats import norm
from tqdm import tqdm
import pandas as pd
import numpy as np
import os
__all__ = ["Background"]
from astropy.cosmology import LambdaCDM

class Background(object):
    """Class to compute cosmological distances."""

    def __init__(self, cosmo=None, interp=False, **kwargs_interp):
        """

        :param cosmo: instance of astropy.cosmology
        :param interp: boolean, if True, uses interpolated cosmology to evaluate specific redshifts
        :param kwargs_interp: keyword arguments of CosmoInterp specifying the interpolation interval and maximum
         redshift
        :return: Background class with instance of astropy.cosmology
        """

        if cosmo is None:
            from astropy.cosmology import default_cosmology

            cosmo = default_cosmology.get()
        if interp:
            self.cosmo = CosmoInterp(cosmo, **kwargs_interp)
        else:
            self.cosmo = cosmo

    @staticmethod
    def a_z(z):
        """Returns scale factor (a_0 = 1) for given redshift.

        :param z: redshift
        :return: scale factor
        """
        return 1.0 / (1 + z)

    def d_xy(self, z_observer, z_source):
        """

        :param z_observer: observer redshift
        :param z_source: source redshift
        :return: angular diameter distance in units of Mpc
        """
        D_xy = self.cosmo.angular_diameter_distance_z1z2(z_observer, z_source)
        return D_xy

    def ddt(self, z_lens, z_source):
        """Time-delay distance.

        :param z_lens: redshift of lens
        :param z_source: redshift of source
        :return: time-delay distance in units of proper Mpc
        """
        return (
            self.d_xy(0, z_lens)
            * self.d_xy(0, z_source)
            / self.d_xy(z_lens, z_source)
            * (1 + z_lens)
        )

    def T_xy(self, z_observer, z_source):
        """

        :param z_observer: observer
        :param z_source: source
        :return: transverse comoving distance in units of Mpc
        """
        D_xy = self.d_xy(z_observer, z_source)
        T_xy = D_xy * (1 + z_source)
        return T_xy

    @property
    def rho_crit(self):
        """Critical density.

        :return: value in M_sol/Mpc^3
        """
        h = self.cosmo.H(0).value / 100.0
        return 3 * h**2 / (8 * np.pi * const.G) * 10**10 * const.Mpc / const.M_sun
    

#From Lenstronomy: https://github.com/lenstronomy/lenstronomy/blob/fd3f872a8802046556d595634256b590a8d58364/lenstronomy/Cosmo/lens_cosmo.py

__author__ = "sibirrer"

# this file contains a class to convert lensing and physical units


__all__ = ["LensCosmo"]


class LensCosmo(object):
    """Class to manage the physical units and distances present in a single plane lens
    with fixed input cosmology."""

    def __init__(self, z_lens, z_source, cosmo=None):
        """

        :param z_lens: redshift of lens
        :param z_source: redshift of source
        :param cosmo: astropy.cosmology instance
        """

        self.z_lens = z_lens
        self.z_source = z_source
        self.background = Background(cosmo=cosmo)
        self.nfw_param = NFWParam(cosmo=cosmo)

    @property
    def h(self):
        return self.background.cosmo.H(0).value / 100.0

    @property
    def dd(self):
        """

        :return: angular diameter distance to the deflector [Mpc]
        """
        return self.background.d_xy(0, self.z_lens)

    @property
    def ds(self):
        """

        :return: angular diameter distance to the source [Mpc]
        """
        return self.background.d_xy(0, self.z_source)

    @property
    def dds(self):
        """

        :return: angular diameter distance from deflector to source [Mpc]
        """
        return self.background.d_xy(self.z_lens, self.z_source)

    @property
    def ddt(self):
        """

        :return: time delay distance [Mpc]
        """
        return (1 + self.z_lens) * self.dd * self.ds / self.dds

    @property
    def sigma_crit(self):
        """Returns the critical projected lensing mass density in units of M_sun/Mpc^2.

        :return: critical projected lensing mass density
        """
        if not hasattr(self, "_sigma_crit_mpc"):
            const_SI = const.c**2 / (
                4 * np.pi * const.G
            )  # c^2/(4*pi*G) in units of [kg/m]
            conversion = const.Mpc / const.M_sun  # converts [kg/m] to [M_sun/Mpc]
            factor = const_SI * conversion  # c^2/(4*pi*G) in units of [M_sun/Mpc]
            self._sigma_crit_mpc = (
                self.ds.value / (self.dd.value * self.dds.value) * factor
            )  # [M_sun/Mpc^2]
        assert self.ds.unit.to_string()=='Mpc' #Assert the units are in Mpc
        assert self.dd.unit.to_string()=='Mpc' #Assert the units are in Mpc
        assert self.dds.unit.to_string()=='Mpc' #Assert the units are in Mpc
        return self._sigma_crit_mpc

    @property
    def sigma_crit_angle(self):
        """Returns the critical surface density in units of M_sun/arcsec^2 (in physical
        solar mass units) when provided a physical mass per physical Mpc^2.

        :return: critical projected mass density
        """
        if not hasattr(self, "_sigma_crit_arcsec"):
            const_SI = const.c**2 / (
                4 * np.pi * const.G
            )  # c^2/(4*pi*G) in units of [kg/m]
            conversion = const.Mpc / const.M_sun  # converts [kg/m] to [M_sun/Mpc]
            factor = const_SI * conversion  # c^2/(4*pi*G) in units of [M_sun/Mpc]
            self._sigma_crit_arcsec = (
                self.ds.value / (self.dd.value * self.dds.value) * factor * (self.dd.value * const.arcsec) ** 2
            )  # [M_sun/arcsec^2]
        assert self.ds.unit.to_string()=='Mpc' #Assert the units are in Mpc
        assert self.dd.unit.to_string()=='Mpc' #Assert the units are in Mpc
        assert self.dds.unit.to_string()=='Mpc' #Assert the units are in Mpc
        return self._sigma_crit_arcsec

    def phys2arcsec_lens(self, phys):
        """Convert physical Mpc into arc seconds.

        :param phys: physical distance [Mpc]
        :return: angular diameter [arcsec]
        """
        assert self.dd.unit.to_string()=='Mpc' #Assert the units are in Mpc
        return phys / self.dd.value / const.arcsec

    def arcsec2phys_lens(self, arcsec):
        """Convert angular to physical quantities for lens plane.

        :param arcsec: angular size at lens plane [arcsec]
        :return: physical size at lens plane [Mpc]
        """
        assert self.dd.unit.to_string()=='Mpc' #Assert the units are in Mpc
        return arcsec * const.arcsec * self.dd.value

    def arcsec2phys_source(self, arcsec):
        """Convert angular to physical quantities for source plane.

        :param arcsec: angular size at source plane [arcsec]
        :return: physical size at source plane [Mpc]
        """
        assert self.ds.unit.to_string()=='Mpc' #Assert the units are in Mpc
        return arcsec * const.arcsec * self.ds.value

    def kappa2proj_mass(self, kappa):
        """Convert convergence to projected mass M_sun/Mpc^2.

        :param kappa: lensing convergence
        :return: projected mass [M_sun/Mpc^2]
        """
        return kappa * self.sigma_crit

    def mass_in_theta_E(self, theta_E):
        """Mass within Einstein radius (area * epsilon crit) [M_sun]

        :param theta_E: Einstein radius [arcsec]
        :return: mass within Einstein radius [M_sun]
        """
        mass = self.arcsec2phys_lens(theta_E) ** 2 * np.pi * self.sigma_crit
        return mass

    def sis_sigma_v2theta_E(self, v_sigma):
        """Converts the velocity dispersion into an Einstein radius for a SIS profile.

        :param v_sigma: velocity dispersion (km/s)
        :return: theta_E (arcsec)
        """
        assert self.dd.unit.to_string()=='Mpc' #Assert the units are in Mpc
        assert self.dds.unit.to_string()=='Mpc' #Assert the units are in Mpc
        theta_E = (
            4
            * np.pi
            * (v_sigma * 1000.0 / const.c) ** 2
            * self.dds.value
            / self.ds.value
            / const.arcsec
        )
        return theta_E