#!/bin/bash

#spectroscopic case
python plot_data_spectroscopic.py #visualise data

python mcmc_standard.py nb #generate unbiased posterior with mcmc
python mcmc_standard.py bi #generate biased posterior with mcmc
python mcmc_spectroscopic.py #generate posterior using zBEAMS on biased data

python plot_contour_spectroscopic.py #plot the 3 posteriors shown above

#photometric case
python mcmc_photometric.py nb 0 #generate unbiased posterior with mcmc (very inefficient**)
python mcmc_photometric.py bi 0 #generate biased posterior with mcmc (very inefficient**)
python mcmc_photometric.py bi 1 #generate posterior using zBEAMS
#** reduce 'n' steps to 50000, set thinning to 1 and burnin to 0 so that code runs much faster.

python process_z_chains.py #post-process all the redshift chains for plots

python plot_data_photometric.py #visualise data

python plot_contour_photometric.py #plot the 3 posteriors shown above

python plot_redshifts_photometric.py #plot 998 redshift histograms