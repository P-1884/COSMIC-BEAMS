import sys
import pandas as pd
from mcmcfunctions_SL import mcmc_SL
from zbeamsfunctions_SL import likelihood_SL,likelihood_spec_contam_SL,likelihood_phot_contam_SL,likelihood_phot_SL,r_SL

argv = sys.argv
print(argv)
filein = argv[1]
contaminated = eval(argv[2])
photometric = eval(argv[3])
print('Contaminated',contaminated,type(contaminated),'Photometric',photometric,type(photometric))
cosmo_type = argv[4]
steps = int(argv[5])

db_in = pd.read_csv(filein)

try:
    walkers = int(argv[6])
except:
    if contaminated and photometric: walkers = int(2*(13+2*len(db_in)))
    elif not contaminated and photometric: walkers = int(2*(9+2*len(db_in)))
    elif contaminated and not photometric: walkers = 40
    elif not contaminated and not photometric: walkers = 40

print(f'Retrieving data from {filein}, assuming the sample is {(not contaminated)*"not "}contaminated and {(not photometric)*"not "}photometric.')
print(f'Am assuming a {cosmo_type} cosmology.')
print(f'Will take {steps} MCMC steps, with {walkers} walkers. Have a dataset of {len(db_in)} systems.')

if not contaminated:
    if not photometric: likelihood = likelihood_SL
    if photometric: likelihood = likelihood_phot_SL
else: 
    if not photometric: likelihood = likelihood_spec_contam_SL
    if photometric: likelihood = likelihood_phot_contam_SL

trunc_r = True #Asserting r>0 in the likelihood, in the case of spectroscopic redshifts and no contamination (a test case)
mcmc_SL(
     n = steps,
     n_walkers = walkers,
     likelihood = likelihood,
     zbias = 'bias',
     mubias = 'bias',
     OMi = 0.3,
     Odei = 0.7,
     H0i = 70,
     wi = -1,
     wai = 0,
     omstep = 0.01,
     ode_step = 0.01,
     H0step = 1,
     wstep = 0.01,
     db_in = db_in, #input file (dataset)
     fileout = f'chains/SL_orig_{filein.split("/")[2]}',
     status = True,
     cosmo_type = cosmo_type,
     contaminated = contaminated,
     photometric = photometric,
     trunc_r=trunc_r)

#addqueue -c '10min' -m 3 -n 5 -s /mnt/users/hollowayp/python11_env/bin/python3.11 ./run_zBEAMS.py
