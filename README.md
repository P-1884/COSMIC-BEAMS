# COSMIC-BEAMS

**COSMIC-BEAMS** (**CO**smology using **S**trong lensing **M**easurements **I**ncluding **C**ontamination – **B**ayesian **E**stimation **A**pplied to **M**ultiple **S**pecies) is a Bayesian framework for inferring cosmological parameters from **photometric and impure samples of strong gravitational lenses**.

The original code was taken from [zBEAMS](https://github.com/MichelleLochner/zBEAMS/tree/master), though has been substantially rewritten for the strong lensing case.

## Code Structure

The repository is organised as follows:

| File | Description |
|------|-------------|
| `run_COSMIC_BEAMS.py` | Main entry point for running COSMIC-BEAMS. Specifies the input data and optional keyword arguments for configuring the likelihood and inference. |
| `mcmcfunctions_SL_JAX_margin_public.py` | Contains the likelihood functions and MCMC sampling routines. |
| `cosmology_JAX_(flat)_public.py` | JAX implementations of the cosmology functions used within the likelihood. |
| `Lenstronomy_Cosmology.py` | Additional cosmology functions used to validate the JAX cosmology implementation. |
| `numpyro_truncnorm_GMM_fit_public.py` | Fits a truncated Gaussian Mixture Model (GMM) to data using NumPyro. |
| `LogNormal_Distribution_Class.py` | JAX implementation of a log-normal probability distribution. |

## Licence

This project is licensed under the [MIT License](https://choosealicense.com/licenses/mit/).

## Citation

If you use COSMIC-BEAMS in your research, please cite:

> Holloway et al. (2026, in preparation)
