import numpy
import random
import sys
from scipy import interpolate
from scipy import integrate
import toy_like
import toy_data
import SN_like
import SN_data
import time
import decimal

#Supernovae case (see mcmc_beams_v2.py for old version with toy model)
#Implements a Monte-Carlo Markov Chain for an arbitrary data set.
#Input: likelihood - log-likelihood function
#       prior - a "function of functions". A function containing the prior (as probability
#               distribution) for each parameter. An integer specifies which prior to use.
#       t0 - the initial vector of parameters to use
#       N   - the number of times to run the loop
#       sig - vector of standard deviations for each parameter ("stepsize")
#       data - 1d vector of data
#       restart - boolean, whether or not to read from file to get first parameters (still provide some arb t0)

#Required constants
c = 2.99792e5 #In km/s
DEC=False #Using the decimal class or not (code speeds up without it)

def mcmc(likelihood,prior,t0,N,sig,data,filename,**kwargs):
    COV=True #Turn the use of the covariance matrix on and off
    LOG=False #Using loglikelihood instead of likelihood
    
    n = len(t0)
    sucsteps = 0 #Number of successful steps

    if kwargs['restart']:
        chain = numpy.loadtxt(filename)
        t0 = chain[-1,:]
    
    else:
        #Output file
        f = open(filename,'w')
        f.close()

    a = time.time()
    like = likelihood(t0,data) #Calculate the log-likelihood with the starting paramters
    b=time.time()
    
    #Use an array in memory for the chain to speed it up
    #CHAIN=numpy.zeros([5000, len(t0)+1])
    CHAIN=numpy.zeros([N, len(t0)+1])
    
    bad = 0 #Number of times likelihood hits infinity

    #If a covariance matrix is given, we diagonalise and adjust step direction accordingly      
    if 'covariance' in kwargs:
        covmat = kwargs['covariance']
        #Perform a Cholesky decomposition
        Q = numpy.linalg.cholesky(covmat)
        

    else:
        covmat = numpy.mat(0)

    for step in range(N):
        if step%1000 == 0:
            print step
            
        if step%10000==0:
            print 'Acceptance ratio:',  100.0*sucsteps/N, '%'

        CHAIN[step, :-1]=t0
        CHAIN[step, -1]=like


#        CHAIN[step%5000, :-1]=t0
#        CHAIN[step%5000, -1]=like
        
#       #Save to file
#        if step==N-1:
#            f=open(filename, 'a')
#            CHAIN=CHAIN[0:(N-1)%5000]
#            numpy.savetxt(f, CHAIN)
#            f.close()
#        if step%5000==0 and step !=0:
#            f=open(filename, 'a')
#            numpy.savetxt(f, CHAIN)
#            f.close()
            
        
#        f = open(filename,'a')
#       #Since the number of parameters is variable, we have to use a little loop
#       #to write to the file
#        for p in range(n):
#            f.write('%0.6e\t' %(t0[p])) 
#        
#        if DEC:
#            f.write(str(like))  #Changed this because we're expecting a decimal object with set precision
#        else:
#            f.write('%0.6e\t' %(like))  
#        f.write('\n')


        #If we've done 2000 steps, or any multiple of, recalculate the covariance matrix
        if step==5000 and COV:
            #Read in the chain
            #chain_all = numpy.loadtxt(filename)
            chain_all=CHAIN[:5000, :]
            #Only take the columns that are being varied
            first = True
            for i in range(n):
                if sig[i] != 0:
                    if first: #First column in the chain
                        chain = chain_all[:,i]
                        first = False
                    else:
                        chain = numpy.column_stack((chain,chain_all[:,i]))
            covmat = covariance(chain)
            #print covmat
            Q = numpy.linalg.cholesky(covmat)
              


        if covmat.size>1:  
            
            #Create a random number vector (delta theta)
            dt = []
            for i in range(len(Q[:,0])):
                #Find delta theta using a random gaussian number
                dt.append(random.gauss(0.0,1.0))
            dt = numpy.array(dt)

            vec = numpy.dot(Q,dt)
            u = []
            j = 0
            for i in range(n):
                if sig[i] != 0:
                    u.append(t0[i]+vec[j])
                    j+=1
                else: #Not varying this parameter
                    u.append(t0[i])
            
        else:
            u = []
            for i in range(n):
                #Find delta theta using a random gaussian number
                u.append(t0[i] + random.gauss(0.0,sig[i]))
       
        #Find the likelihood value for the new parameters.
        P1=prior(t0, u)
        if P1==0:
            #print 'prior=0 at', step
            if DEC:
              like_u=decimal.Decimal(0)
            else:
              like_u=0.0
        else:
            like_u = likelihood(u,data)
           

        #Use the priors to weight the likelihood
        #print P1
        
        #For some reason, sometimes a jump is so big R gets so high it is infinity to Python. 
        #The safest is to set R=1 (obviously it's a better step). This should be very rare 
        #for small step sizes
        """if numpy.isinf(numpy.exp(float(like_u-like))) and LOG: #Floats specified to convert Decimal class
            R = 1.0
            bad+=1
            print 'bad'
        elif numpy.isinf(float(like_u/like)) and LOG==False:
            R = 1.0
            bad+=1
            print 'bad'
        else:"""
        #Finally calculate the ratio of probabilities between the old and new step
        if LOG:
            R = P1*float(numpy.exp(like_u-like))
        else:
            if DEC:
              R = decimal.Decimal((str)(P1))*(like_u/like)
            else:
              R = P1*(like_u/like)
        #print like_u/like

        #Sometimes, the likelihood does crazy things (like if p>1). If P1 is zero, R should simply be set to zero
       # if P1==0:
        #    R = 0
        
        #This only applies if we're not using log likelihoods
        """ if LOG==False:
            #If R>1, accept this step. Still accept this step with the probability R if R<1
            #We need to deal with the possibility that the likelihood comes out negative (from p-BEAMS) so we just
            #always accept a step with a larger likelihood
            if like_u>like:
                R=1*P1 
            else:
                if like_u<0 and like<0: #Deals with the case they are both negative and the new step is worse
                    R=P1*(float)(like/like_u)
                #else:
                    #R is fine"""
        if DEC:
          r = decimal.Decimal((str)(random.random()))
        else:
          r = random.random()
        #print like_u
        #Step is accepted
        if r<=R:
            t0 = u
            like = like_u
            sucsteps += 1
           # print 'accept'
        #Else, step is rejected and t0 remains
        f.close()
    numpy.savetxt(filename, CHAIN)
    A = 100.0*sucsteps/N
    print 'Acceptance ratio: ',A
    #print 'bad steps: ',bad





#Given a chain (or subset thereoff), calculates the covariance matrix (likelihood removed)
def covariance(chain):
    n = len(chain[0,:])
    N = len(chain[:,0])
    #Find the means
    mu = numpy.mean(chain,axis=0)

    cov = numpy.zeros([n,n])
    for i in range(n):
        for j in range(n):
            #Sum over all values in the data range
            total = 0.0
            for k in range(N):
                total += (chain[k,i]-mu[i]) * (chain[k,j]-mu[j])
            cov[i,j] = total/N
    return cov
            


#Function to extract the parameter values from a mock dataset
def get_params(filename):
    f=open(filename, 'r')
    for line in f.readlines():
        if line.lstrip().find('sig1') !=-1:
            sig1=(float)(line.lstrip()[len('sig1 = '):])
        elif line.lstrip().find('sig2')!=-1:
            sig2=(float)(line.lstrip()[len('sig2 = '):])
        elif line.lstrip().find('b')!=-1:
            b=(float)(line.lstrip()[len('b = '):])
        elif line.lstrip().find('x')!=-1:
            x=(float)(line.lstrip()[len('x = '):])
        elif line.lstrip().find('y')!=-1:
            y=(float)(line.lstrip()[len('y = '):])
        elif line.lstrip().find('z')!=-1:
            z=(float)(line.lstrip()[len('z = '):])
    f.close()
    return (sig1, sig2, b)
    
#Function to extract the parameter values from a mock dataset
def get_params_toy(filename):
    f=open(filename, 'r')
    for line in f.readlines():
        if line.lstrip().find('the1') !=-1:
            the1=(float)(line.lstrip()[len('the1 = '):])
        elif line.lstrip().find('the2')!=-1:
            the2=(float)(line.lstrip()[len('the2 = '):])
        elif line.lstrip().find('sig1') !=-1:
            sig1=(float)(line.lstrip()[len('sig1 = '):])
        elif line.lstrip().find('sig2')!=-1:
            sig2=(float)(line.lstrip()[len('sig2 = '):])
        elif line.lstrip().find('x')!=-1:
            x=(float)(line.lstrip()[len('x = '):])
        elif line.lstrip().find('y')!=-1:
            y=(float)(line.lstrip()[len('y = '):])
        elif line.lstrip().find('z')!=-1:
            z=(float)(line.lstrip()[len('z = '):])
    f.close()
    return (the1,the2,sig1, sig2, x, y, z)

############################## Main ########################################
#Use system arguments for flags. First argument must be the number of the chain,
#the second must be 'correlated' or 'uncorrelated'. The restart flag is optional

RES = False

num = sys.argv[1]
root=sys.argv[2]

if 'restart' in sys.argv:
    #Flag for restarting the chain
    RES = True
    
    
if 'toy' in sys.argv:
    [the1,the2,sig1,sig2,x,y,z]=get_params_toy(root+'data_parameters.txt')
    P_t1=toy_like.P_t1
    P_t2=toy_like.P_t2
    P_t3=toy_like.P_t3
    P_s1=toy_like.P_s1
    P_s2=toy_like.P_s2
    P_s3=toy_like.P_s3
    P_a=toy_like.P_a
    #x=0.5
    #y=0.00
    #z = 0.5 #Trying some more intense correlations
        
    N=100000

    
    if 'correlated' in sys.argv:
        #data_name = 'data_cor_toy.txt'
        #root = 'test7/'
        data_name = root+'data_cor_toy.txt'
        data = numpy.loadtxt(data_name)
        
        
        
        t0 = [random.random()*20+40,random.random()*20+20,sig1,sig2,x, y, z]
        
        sig = [2,2,0,0,0,0,0]
        if 'perturbed' in sys.argv:
            filename=root+'chain_toy_p_beams_cor_%s.txt' %(num)
            mcmc(toy_like.perturbed_cor_like,toy_like.prior,t0,N,sig,data,filename,restart=RES)
        else:
            filename=root+'chain_toy_beams_cor_%s.txt' %(num)
            mcmc(toy_like.beams_uncor_like,toy_like.prior,t0,N,sig,data,filename,restart=RES)
    else:
        data_name = root+'data_uncor_toy.txt'
        data = numpy.loadtxt(data_name)
        t0 = [random.random()*(P_t1[1]-P_t1[0])+P_t1[0],random.random() *(P_t2[1]-P_t2[0])+P_t2[0] ,random.random()*(P_s1[1]-P_s1[0])+P_s1[0], random.random()*(P_s2[1]-P_s2[0])+P_s2[0]]
        #t0 = [random.random()*(P_t1[1]-P_t1[0])+P_t1[0],random.random() *(P_t2[1]-P_t2[0])+P_t2[0] , numpy.log(sig1),numpy.log(sig2)]
        sig = [2,2,0.1,0.1]
        #sig=[2,2,0,0]
        if 'perturbed' in sys.argv:
            filename='toy uncor 5/chain_toy_p_beams_uncor_%s.txt' %(num)
            mcmc(toy_like.perturbed_uncor_like,toy_like.prior,t0,N,sig,data,filename,restart=RES)
        elif 'broad' in sys.argv:
            #Uses a broad likelihood function
            t0 = [random.random()*(P_t1[1]-P_t1[0])+P_t1[0],random.random() *(P_t2[1]-P_t2[0])+P_t2[0] ,random.random()*(P_s1[1]-P_s1[0])+P_s1[0], random.random()*(P_a[1]-P_a[0])+P_a[0]]
            sig = [2,0,0.1,0.1]
            filename=root+'chain_toy_broad_uncor_%s.txt' %(num)
            mcmc(toy_like.beams_uncor_like_broad,toy_like.prior_broad,t0,N,sig,data,filename,restart=RES)
        elif 'three' in sys.argv:
            t0 = [random.random()*(P_t1[1]-P_t1[0])+P_t1[0],random.random() *(P_t2[1]-P_t2[0])+P_t2[0] ,random.random() *(P_t3[1]-P_t3[0])+P_t3[0] ,random.random()*(P_s1[1]-P_s1[0])+P_s1[0], random.random()*(P_s2[1]-P_s2[0])+P_s2[0], random.random()*(P_s3[1]-P_s3[0])+P_s3[0] , 1, 1]
            sig = [2,2, 2,0.1,0.1,0.1, 1, 1 ]
            filename=root+'chain_toy_three_uncor_%s.txt' %(num)
            mcmc(toy_like.beams_uncor_like_3,toy_like.prior_3,t0,N,sig,data,filename,restart=RES)
        else:
            filename=root+'chain_toy_beams_uncor_%s.txt' %(num)
            mcmc(toy_like.beams_uncor_like,toy_like.prior,t0,N,sig,data,filename,restart=RES)
else:
    #root_main='/space/knights/big_sim2/'
    
    a=time.time()
    N=100000
    #The SN model
    
    #Actual cosmology
    H0_t = SN_data.H0 #In km/s/Mpc
    om_t = SN_data.om_m
    ol_t = SN_data.om_l
    b=SN_data.b
    sig1=SN_data.sig1
    sig2=SN_data.sig2

    """sig1 = SN_data.sig1 #Std dev of 1a's
    sig2 = SN_data.sig2 #Std dev of non-1a's
    b = SN_data.b #shift    
    x=SN_data.x
    y=SN_data.y
    z = SN_data.z"""
    
    P_om = SN_like.P_om
    P_ol = SN_like.P_ol
    P_H0 = SN_like.P_H0
    P_sig1=SN_like.P_sig1
    P_sig2=SN_like.P_sig2
    P_b=SN_like.P_b
    P_x = SN_like.P_x
    P_S = SN_like.P_S #The prior on s1-s5.
    P_Sb=SN_like.P_Sb

    
    if 'correlated' in sys.argv:
        #root = 'SN_cor_5_v2/'
        #root = 'SN_cor_marg_200_v2/'
        #root = 'SN_cor_marg_200/'
        data_name = root+'SN_cor.txt'

        data = numpy.loadtxt(data_name)
        #t0 = [numpy.random.random()*(0.5-0.05)+0.05,numpy.random.random()*(0.9-0.5)+0.5,numpy.random.random()*(80-60)+60,  b, sig1, sig2, x, y, z] 
        #t0 = [numpy.random.random()*(0.5-0.05)+0.05,numpy.random.random()*(0.9-0.5)+0.5,H0,  b, sig1, sig2, x, y, z] 
        #t0 = [om_m,om_l, H0, b, sig1, sig2, x, y, z] 
        #print 't0 ', t0
        #sig=[0.1, 0.1, 2, 0, 0, 0, 0, 0, 0]
        
        #Try starting closer to the right values
        """om_m = random.random()*(P_om[1]-P_om[0])+P_om[0]
        om_l = random.random()*(P_ol[1]-P_ol[0])+P_ol[0]
        z_max = 1.5
        while om_m*(1+z_max)**3.0+om_l+(1-om_m-om_l)*(1+z_max)**2.0<0:
            om_m = random.random()*(P_om[1]-P_om[0])+P_om[0]
            om_l = random.random()*(P_ol[1]-P_ol[0])+P_ol[0]
        t0 = [om_m, om_l,random.random()*(P_H0[1]-P_H0[0])+P_H0[0],  b, sig1, sig2,   x, y, z] """
        
        om_m = random.random()*(P_om[1]-P_om[0])+P_om[0]
        om_l = random.random()*(P_ol[1]-P_ol[0])+P_ol[0]
        z_max = 2.0
        while om_m*(1+z_max)**3.0+om_l+(1-om_m-om_l)*(1+z_max)**2.0<0:
            om_m = random.random()*(P_om[1]-P_om[0])+P_om[0]
            om_l = random.random()*(P_ol[1]-P_ol[0])+P_ol[0]
            
        h0= random.random()* (P_H0[1]-P_H0[0])+P_H0[0]
        b_new= random.random()* (P_b[1]-P_b[0])+P_b[0]
        s1= random.random()* (P_sig1[1]-P_sig1[0])+P_sig1[0]
        s2= random.random()* (P_sig2[1]-P_sig2[0])+P_sig2[0]
        
        #t0 = [om_m, om_l,h0,b_new,s1,s2,  -1, -1, -1,-1,-1]
        #t0 = [om_m, om_l,h0,b,numpy.log(sig1),numpy.log(sig2),   -1, -1, -1, -1,-1]
        #t0 = [om_m, om_l,h0,b_new,s1,s2,   -1, -1, -1,-1,-1]
        t0 = [om_t, ol_t,H0_t,b,numpy.log(sig1),numpy.log(sig2),  -1, -1, -1,-1,-1]
        sig=[0.1, 0.1, 2.0, 0.1, 0.1, 0.1, 0.0, 0.0, 0.0,0,0]
        #sig=[0.1, 0.1, 2.0, 0., 0.1, 0.05, 0.0, 0.0, 0.0,0,0]
        #sig=[0.1, 0.1, 2.0, 0., 0., 0.0, 0.0, 0.0, 0.0,0,0]
        #sig=[0.1, 0.1, 2.0, 0., 0.1, 0.05, 0.0, 0.0, 0.0,0,0]
        #sig=[0.1, 0.1, 2.0, 0.1, 0.01, 0.1, 0.0, 0.0, 0.0,0,0] #Not using logs
        
        #t0 = [om_m,om_l, H0, b, sig1, sig2, x, y, z] 
        
        if 'perturbed' in sys.argv:
            filename=root+'chain_SN_p_beams_cor_%s.txt' %(num)
            mcmc(SN_like.perturbed_cor_like,SN_like.prior,t0,N,sig,data,filename,restart=RES)
        elif 'true' in sys.argv:
            filename=root+'chain_SN_true_beams_cor_%s.txt' %(num)
            mcmc(SN_like.true_beams,SN_like.prior,t0,N,sig,data,filename,restart=RES)
        #Jackknifing of true BEAMS. Will remove the 'num'th data point and label the chain appropriately
        elif 'jackknife' in sys.argv:
            data = numpy.vstack((data[:(int)(num), :], data[(int)(num)+1:, :]))
            filename=root+'chain_SN_jackknife_cor_%s.txt' %(num)
            mcmc(SN_like.true_beams,SN_like.prior,t0,N,sig,data,filename,restart=RES)
        elif 'special' in sys.argv:
            filename=root+'chain_SN_sp_p_beams_cor_%s.txt' %(num)
            mcmc(SN_like.special_perturbed_cor_like,SN_like.prior,t0,N,sig,data,filename,restart=RES)
        elif 'chi2' in sys.argv:
            #filename='example/chain_SN_beams_1a_%s.txt' %(num)
            t0 = [om_m, om_l,h0,b,s1,numpy.log(sig2), 0, 1.5, 1.5, 1.5, 1.5]
            sig=[0.1, 0.1, 2.0, 0., 0.1, 0., 0.0, 0.0, 0.0,0,0]
            filename=root+'chain_chi2_%s.txt' %(num)
            data=data[data[:, 2]>0.9, :]
            mcmc(SN_like.chi2_cor,SN_like.prior,t0,N,sig,data,filename,restart=RES)
        else:
            filename=root+'chain_SN_beams_cor_%s.txt' %(num)
            if DEC:
                like=SN_like.beams_uncor_like_DEC
            else:
                like=SN_like.beams_uncor_like
            mcmc(like,SN_like.prior,t0,N,sig,data,filename,restart=RES)
    
    elif 'renee' in sys.argv:
        data_name = root+'renee.txt' #Testing correlations
        a0=1.5
        a1=1.0
        a2=-3.0
        data = numpy.loadtxt(data_name)
        
        om_m = random.random()*(P_om[1]-P_om[0])+P_om[0]
        om_l = random.random()*(P_ol[1]-P_ol[0])+P_ol[0]
        z_max = 1.5
        while om_m*(1+z_max)**3.0+om_l+(1-om_m-om_l)*(1+z_max)**2.0<0:
            om_m = random.random()*(P_om[1]-P_om[0])+P_om[0]
            om_l = random.random()*(P_ol[1]-P_ol[0])+P_ol[0]
            
        lgs1 = random.random()*(P_sig1[1]-P_sig1[0])+P_sig1[0]
        lgs2 = random.random()*(P_sig2[1]-P_sig2[0])+P_sig2[0]
        h0=random.random()*(P_H0[1]-P_H0[0])+P_H0[0]
        
        t0 = [om_m, om_l,h0, a0, a1, a2, lgs1, lgs2]  
        sig=[0.1, 0.1, 2.0, 0, 0, 0, 0.5, 0.5]
        
        filename=root+'chain_gaussian_uncor_%s.txt' %(num)
        mcmc(SN_like.beams_renee,SN_like.prior_renee,t0,N,sig,data,filename,restart=RES)
    
    else:
        #root = 'SN_p_uncor_200/'
        #root = 'SN_uncor_marg_200_v3/'
        #root='SN_uncor_marg_1000/'
        data_name = root+'SN_uncor.txt' #Testing correlations
        #sig1, sig2, b = get_params(root+'data_parameters.txt')
        sig1, sig2, b=0.1, 1.5, 2.0
        sig1=numpy.log(sig1)
        sig2=numpy.log(sig2)
        data = numpy.loadtxt(data_name)
        #t0 = [numpy.random.random()*(0.5-0.05)+0.05,numpy.random.random()*(0.9-0.5)+0.5,numpy.random.random()*(80-60)+60,  b, sig1, sig2, x, y, z] 
        #sig2=1.1255407617422655
        #sig1=0.96067501827816371
        om_m = random.random()*(P_om[1]-P_om[0])+P_om[0]
        om_l = random.random()*(P_ol[1]-P_ol[0])+P_ol[0]
        z_max = 1.5
        while om_m*(1+z_max)**3.0+om_l+(1-om_m-om_l)*(1+z_max)**2.0<0:
            om_m = random.random()*(P_om[1]-P_om[0])+P_om[0]
            om_l = random.random()*(P_ol[1]-P_ol[0])+P_ol[0]
        h0= random.random()* (P_H0[1]-P_H0[0])+P_H0[0]
        b_new= random.random()* (P_b[1]-P_b[0])+P_b[0]
        s1= random.random()* (P_sig1[1]-P_sig1[0])+P_sig1[0]
        s2= random.random()* (P_sig2[1]-P_sig2[0])+P_sig2[0]
        
        t0 = [om_m, om_l,h0,b_new,s1,s2]
        sig=[0.1, 0.1, 2.0, 0.1, 0.1, 0.1]

        #t0 = [om_m,om_l, H0, b, sig1, sig2, x, y, z] 

        #if 'perturbed' in sys.argv:
        if 'beams' in sys.argv:
            filename=root+'chain_SN_beams_uncor_%s.txt' %(num)
            #mcmc(SN_like.perturbed_uncor_like,SN_like.prior,t0,N,sig,data,filename,restart=RES)
            mcmc(SN_like.beams_uncor_like,SN_like.prior,t0,N,sig,data,filename,restart=RES)
        elif 'perturbed' in sys.argv:
            filename=root+'chain_SN_p_beams_uncor_%s.txt' %(num)
            #mcmc(SN_like.perturbed_uncor_like,SN_like.prior,t0,N,sig,data,filename,restart=RES)
            mcmc(SN_like.perturbed_uncor_like,SN_like.prior,t0,N,sig,data,filename,restart=RES)
        else:
            #filename='example/chain_SN_beams_1a_%s.txt' %(num)
            filename=root+'chain_chi2_%s.txt' %(num)
            mcmc(SN_like.chi2,SN_like.prior,t0,N,sig,data,filename,restart=RES)
    b=time.time()
    print 'Time taken:',(b-a)/60.0,'min'



