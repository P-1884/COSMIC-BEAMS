from __future__ import division, print_function
import sys
from numpy import *

#Code to calculate the Gelman-Rubin statistic from a set of chains
#Input: chains - a list of mcmc chains (post burn-in)
# p - the parameter index 
def gelman(chains,p):
    M = len(chains)
    #We first need to prune the chains to the same length
    comp=[]
    for i in range(M):
        comp.append(len(chains[i][:, 0]))
    n = min(comp)

    for i in range(M):
        chains[i] = chains[i][:n, :]

    means = [] #Contains the mean for each chain
    variances = [] #Contains the variance for each chain
    for m in range(M):
        means.append(mean(chains[m][:, p]))
        variances.append(std(chains[m][:, p])**2)
            
    means=array(means)
    variances = array(variances)
    chain_mean = mean(means) #Get the between-chain mean
    
    B = n/(float)(M-1)*sum((means-chain_mean)**2) #Between-chain variance
    W = 1.0/M*sum(variances) #Within chain variance

    V = (float)(n-1)/n*W + (float)(M+1)/n/M*B #Posterior marginal variance
    
    return sqrt(V/W)
    
#Find the least number of steps that all chains have converged for all parameters
def converge(root, chainname, burn, nchains):
    chains=[]
    #burn=10000 #Burn-in
    for i in range(nchains):
    #for i in [0,2]:
        print(i)
        #chains.append(loadtxt(root+ chainname %(i,i))[0::2, :])
        chains.append(loadtxt(root+ chainname %(i))[:, :])
    p = range(len(chains[0][0, :-1]))
    #Jump in steps of 5000 from the burn-in
    jump=5000
    for step in range(burn, 200000, jump):
        print(step)
        chains_new = []
        gel = []
        for i in p:
            for j in range(len(chains)):
#                print len(chains[j][:,0])
                if step>len(chains[j][:, 0]):
                    return -1 #Convergence was not reached before the end of the chain
                chains_new.append(chains[j][:step, :])
                

            gel.append(gelman(chains_new, i))
        
        print(gel)
        if max(gel)<1.01:
            #return step #We have converged
            break
            
    #Now we know at what step to start with, now we can calculate at exactly what point we converged
    for s in range(step-jump,  step, 200):
        chains_new = []
        gel = []
        for i in p:
            for j in range(len(chains)):
                chains_new.append(chains[j][:s, :])
            gel.append(gelman(chains_new, i))
        if max(gel)<1.01:
            return s #We have converged
            
#Find the least number of steps that all chains have converged for all parameters
def converge_from_list(chains, jump=500):
    
    nchains=len(chains)
    p = range(len(chains[0][0, :]))
    #Jump in steps of 5000 from the burn-in
    #jump=5000
    for step in range(jump, 200000, jump):
        #print(step)
        chains_new = []
        for j in range(len(chains)):
            #print len(chains[j][:,0])
            if step>len(chains[j][:, 0]):
                return -1 #Convergence was not reached before the end of the chain
            chains_new.append(chains[j][:step, :])

        gel = []
        for i in p:
            gel.append(gelman(chains_new, i))
        
        #print gel
        if max(gel)<1.01:
            #return step #We have converged
            break
            
    #Now we know at what step to start with, now we can calculate at exactly what point we converged
    dx=jump//10
    startpoint=step-jump+dx
    for s in range(startpoint,  step, dx):
        #print(s)
        chains_new = []
        gel = []
        for j in range(len(chains)):
            chains_new.append(chains[j][:s, :])
        for i in p:
            gel.append(gelman(chains_new, i))
        if max(gel)<1.01:
            print('Gelman-Rubin coefficients', gel)
            return s+startpoint #We have converged
            
    

#print converge('SN_cor_marg_200_x2/', 'chain_SN_marg_beams_cor_%d.txt',60000)
#print converge('SN_uncor_marg_1000/', 'chain_SN_marg_beams_uncor_%d.txt',30000)     
#print converge('SN_p_uncor_200/', 'chain_SN_beams_uncor_%d.txt',10000)   
#print converge('mock2_smooth/','chain%d/hod_chain_%d.txt',40000)    
#print converge('charles_fake_smooth/','chain%d/hod_chain_%d.txt',40000)
#root=sys.argv[1]
#print converge(root,'chain%d/hod_chain_%d.txt',5000)

#Read in the chains and calculate the gelman-rubin statistic.
#chains=[]
#root = 'SN_uncor_marg_200_v2/'
#burn=10000 #Burn-in
#p=0
#for i in range(4):
#    chains.append(loadtxt(root+'chain_SN_beams_uncor_%d.txt' %(i))[burn:, :])
#    
#print gelman(chains, p)
    
    
    
    
    
