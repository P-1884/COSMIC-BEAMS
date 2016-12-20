#Code to generate some random supernova data
import numpy
from scipy import integrate
from scipy import interpolate
import sys
import os
import matplotlib
import time
#matplotlib.use('GTKAgg')

from matplotlib.colors import LogNorm

rc = matplotlib.rcParams

rc['text.usetex'] = True
rc['font.family']='serif'
rc['font.serif'].insert(0,'Times')
rc['font.size'] = 20
rc['xtick.labelsize']='small'
rc['ytick.labelsize']='small'
rc['legend.fontsize']=19
import pylab

#Some constants
H0 = 70.4 #In km/s/Mpc
om_m = 0.272
om_l = 1-om_m
#om_l = 0.8
c = 2.99792e5 #In km/s

TYPE='block'
global cov_block #This is the covariance matrix from real SNe data
cov_block=[]

#This function takes the number of data points to create, the standard deviations
#of the 1a and contaminating populations, b (shift of contaminating distribution), mean and 
#standard deviation of redshifts and epsilon (how much the probabilities can deviate from 0 and 1).
#It produces a matrix of the distance modulus, (assuming LCDM), redshift and probabilities.
#If Ia=True, we only produce Ia's.
def sn_uncor(N,sig1,sig2,b,z_min,z_max,eps, Ia):
    #First we spline the distance modulus function, cos it takes to long to integrate
    z_spl = numpy.linspace(z_min,z_max,100)
    y_spl = save_dist(z_spl)

    spline = interpolate.splrep(z_spl,y_spl)
    

    #Make the redshifts
    red = []
    for i in range(N):
        #For now I'm artificially using a uniform distribution
        z = numpy.random.random()*(z_max-z_min)+z_min
        red.append(z)
    red = numpy.sort(red)

    prob=[] #Probabilities
    mod=[]  #Distance moduli
    types=[] #Type 1 is 1a, 0 is non-1a
    
    for i in range(N):
        #print i
        #We make sure that our probability lies either between 0 and eps or (1-eps) and 1
        p=numpy.random.random()
        while(p>=eps and p<=(1.0-eps)):
            #We allow about 10% of the data to lie outside this range
            r=numpy.random.random()
            if r<0.05:
                p=numpy.random.random()*(1-eps-eps)+eps
                break
            p = numpy.random.random()
        if Ia:
            p=1

        prob.append(p)

        #Now we assign the type
        r = numpy.random.random()
        if r<p:
            t=1
        else:
            t=0
        #Mess up the first 2 so we're guarenteed to have some wrong ones
        #if i==0 or i==1:
        #    t = 1-t
        types.append(t)

        #Find the mean of the distance modulus from the spline
        m = interpolate.splev(red[i],spline)

        if t==1: #This is a 1a
            mod.append(numpy.random.randn()*sig1+m)
            #mod.append(m)
        else:
            mod.append(numpy.random.randn()*sig2+m+b)
            #mod.append(m+b)
    #print numpy.column_stack((red,mod,prob,types))
    return numpy.column_stack((red,mod,prob,types))



#This function takes the number of data points to create, the standard deviations
#of the 1a and contaminating populations, b (shift of contaminating distribution), mean and 
#standard deviation of redshifts and epsilon (how much the probabilities can deviate from 0 and 1).
#It produces a matrix of the distance modulus, (assuming LCDM), redshift and probabilities.
#X,Y and Z are the coefficients of the covariance matrix
#def sn_cor(N,sig1,sig2,b,x, y, z, z_min,z_max,eps):
def sn_cor(N,sig1,sig2,b,s1, s2, s3, s4, s5, z_min,z_max,eps):
    NEW=True
    #First we spline the distance modulus function, cos it takes to long to integrate
    z_spl = numpy.linspace(z_min,z_max,100)
    y_spl = save_dist(z_spl)

    spline = interpolate.splrep(z_spl,y_spl)
    
    
    #Make the redshifts
    red = []
    for i in range(N):
        #For now I'm artificially using a uniform distribution
        Z = numpy.random.random()*(z_max-z_min)+z_min
        red.append(Z)
    red = numpy.sort(red)
    if NEW:
        numpy.savetxt('red.txt', red)
    else:
        red=numpy.loadtxt('red.txt')

    prob=[] #Probabilities
    mod=[]  #Distance moduli
    types=[] #Type 1 is 1a, 0 is non-1a
    
    for i in range(N):
        #print i
        #We make sure that our probability lies either between 0 and eps or (1-eps) and 1
        p=numpy.random.random()
        while(p>=eps and p<=(1.0-eps)):
            #We allow about 10% of the data to lie outside this range
            r=numpy.random.random()
            if r<0.05:
                p=numpy.random.random()*(1-eps-eps)+eps
                break
            p = numpy.random.random()
        #p=1.0 #Making all objects 1a's
        prob.append(p) 


        #Now we assign the type
        r = numpy.random.random()
        if r<p:
            t=1
        else:
            t=0
        #if i==0 or i==1 or i==2:
         #   t=1-t #Mess it up so that 2 types are wrong
        types.append(t)
    if NEW:
        numpy.savetxt('types.txt', types)
        numpy.savetxt('p.txt', prob)
    else:
        prob=numpy.loadtxt('p.txt')
        types = numpy.loadtxt('types.txt')
    
    #t0 = [om_m,om_l, H0, b, sig1, sig2, x, y, z] 
    t0 = [om_m,om_l, H0, b, sig1, sig2, s1, s2, s3, s4, s5] 
    
    if TYPE=='wed':
        C=wed_cov_mat(types, t0)
    elif TYPE=='block':
        C=block_cov_mat(types, t0, numpy.column_stack((red, types)))
    elif TYPE=='decay':
        C=decay_cov_mat(types, t0)
    #print numpy.linalg.cholesky(C)
    print 'det',numpy.linalg.det(C)
    #print numpy.linalg.slogdet(C)
    numpy.savetxt('cov.txt', C)

    #Construct some residuals to correlate
    res = []
    for i in range(N):
        if types[i]==1: #This is a 1a
            if NEW:
                #res.append(numpy.random.randn()*sig1)
                res.append(0)
            else:
                res.append(0)
        else:
            if NEW:
                #res.append(numpy.random.randn()*sig2)
                res.append(0)
            else:
                res.append(0)
    res = numpy.mat(res).T
    if NEW:
        eta = numpy.mat(numpy.random.randn(N,1))
        numpy.savetxt('eta.txt', eta)
        numpy.savetxt('res.txt', res)
    else:
        eta=numpy.mat(numpy.loadtxt('eta.txt')).T
        res=numpy.mat(numpy.loadtxt('res.txt')).T
    Q = numpy.linalg.cholesky(C)
    
    #print eta
    #print Q
    res_new = res + Q*eta
    #print res_new
    types = numpy.array(types)
    #print numpy.std(res_new[types==1])
    #print numpy.std(res_new[types==0])
    
    for i in range(N):
        #Find the mean of the distance modulus from the spline
        m = interpolate.splev(red[i],spline)
        if types[i]==1:
            mod.append(m+(float)(res_new[i]))
        else:
            mod.append(m+(float)(res_new[i])+b)
    #print mod

   # print numpy.column_stack((red,mod,prob,types))
    #print 'miss-typed: ', sum(abs(numpy.array(prob).round()-types))
    return numpy.column_stack((red,mod,prob,types))


#Little function to give the "distance" factor we use in the covariance matrix
#This is just to make it easier to change across all functions
def factor(i, j):
    #return numpy.exp(abs(i-j))
    return abs(i-j)



def block_cov_mat(types, t, data):
    global cov_block
    types=numpy.array(types)
    N=len(data[:, 0])
    if len(cov_block)==0:
        cov_block=numpy.zeros([N, N])
        c=numpy.loadtxt('SNe_Binned_Covariance.txt')
        bins=[0]*12
        for i in range(1, len(bins)):
            ind=numpy.argmax(data[data[:, 0]<=(float)(i)/10.0, 0])
            bins[i]=ind
        bins[-1]=len(data[:, 0])

        for i in range(len(c[:, 0])):
            for j in range(len(c[0, :])):
                B1=numpy.zeros([N, N])
                B2=numpy.zeros([N, N])
                B1[bins[i]:, bins[j]:bins[j+1]]=1
                B2[bins[i]:bins[i+1], bins[j]:]=1
                B=B1*B2
                B=numpy.array(B, dtype='bool')

                cov_block+=B*c[i, j]

        
    #numpy.set_printoptions(threshold=numpy.nan)
    
    cov_types=numpy.zeros([N, N])
    B=numpy.mat(numpy.array(types==1))
    cov_types[B.T*B]=1
    
    C=cov_types*cov_block
    sigs=numpy.ones([N])*sig1*sig1
    sigs[types==0]=sig2*sig2
    C[range(N), range(N)] += sigs
    
    return numpy.mat(C)

"""#BACKUP
#Block diagonal covariance matrix
def block_cov_mat(types, t):
    sig1 = t[4]
    sig2 = t[5]
    s1 = t[6]
    s2 = t[7]
    s3 = t[8]   
    s4 = t[9]   
    s5=t[10]
    
    errs=[s1, s2, s3, s4, s5]
    
    N=len(types)
    C = numpy.zeros([N,N])
    
    for i in range(N):
        for j in range(i, N):
            if i==j: #The diagonal is just the error, depending on the type
                if types[i]==1:
                    C[i,j] = sig1*sig1
                else:
                    C[i,j] = sig2*sig2
            else:
                #Correlate all elements within a certain distance of the diagonal
                #if abs(i-j)<N/10:
                if types[i]==1 and types[j]==1:
                    err=sig1*sig1
                    #err=1
                elif types[i]==0 and types[j]==0:
                    #err=sig2*sig2
                    err=0
                else:
                    #err=sig1*sig2
                    err=0

                if i<N/5 and j<N/5:
                    C[i,j] = s1*err
                    C[j,i] = s1*err
                elif i<2*N/5 and j<2*N/5 and i>=N/5 and j>=N/5:
                    C[i,j] = s2*err
                    C[j,i] = s2*err
                elif i<3*N/5 and j<3*N/5 and i>=2*N/5 and j>=2*N/5:
                    C[i,j] = s3*err
                    C[j,i] = s3*err
                elif i<4*N/5 and j<4*N/5 and i>=3*N/5 and j>=3*N/5:
                    C[i,j] = s4*err 
                    C[j,i] = s4*err   
                elif i>=4*N/5 and j>=4*N/5:
                    C[i,j] = s5*err
                    C[j,i] = s5*err
                        
    Cr = numpy.mat(C)
    print Cr
    return Cr"""
    
#Covariance matrix from Kim et al.
#We divide the data into 5 blocks. The matrix is of the form of V_ab in Kim et al. and a new error
#contribution is added in each block.
def wed_cov_mat(types, t):

    sig1 = t[4]
    sig2 = t[5]
    s1 = t[6]
    s2 = t[7]
    s3 = t[8]   
    s4 = t[9]   
    s5=t[10]

    errs=[s1, s2, s3, s4, s5]
    #print errs
    
    N=len(types)
    C_tot=numpy.zeros([N,N])
    
    types=numpy.array(types)

    s=0
    blk=N/5 #This causes problems if N/5 isn't already integer, when you multiply int*N/5, you don't get int*(N/5)! Stupid python!
    for i in range(5):
        
        s+=errs[i]

        C = numpy.zeros([N,N])
        
        B=numpy.mat(types==1)
        C[B.T*B]=sig1*sig1*s
 

        sigs=numpy.ones([N])*sig1*sig1*(s+1)
        sigs[numpy.array(types==0)]=sig2*sig2
        #print sigs
        C[range(N), range(N)]=sigs

        #Make the blocks
        B1=numpy.zeros([N, N])
        B2=numpy.zeros([N, N])
        
        if i==4:
            B1[i*blk:, i*blk:]=1
            B2[i*blk:, i*blk:]=1
        else:
            B1[i*blk:, i*blk:i*blk+blk]=1
            B2[i*blk:i*blk+blk, i*blk:]=1

        B=B1+B2
        B=numpy.array(B, dtype='bool')
        #numpy.set_printoptions(threshold=numpy.nan)
        #print B

        C_tot+=C*B

  
    Cr = numpy.mat(C_tot)
    return Cr

"""#BACKUP
#Covariance matrix from Kim et al.
#We divide the data into 5 blocks. The matrix is of the form of V_ab in Kim et al. and a new error
#contribution is added in each block.
def wed_cov_mat(types, t):
    sig1 = t[4]
    sig2 = t[5]
    s1 = t[6]
    s2 = t[7]
    s3 = t[8]   
    s4 = t[9]   
    s5=t[10]
    
    errs=[s1, s2, s3, s4, s5]
    #print errs
    
    N=len(types)
    C = numpy.zeros([N,N])
    
    for i in range(N):
        for j in range(i, N):
            block = min(i, j)/(N/5)
            
            s=errs[0]
            for k in range(block):
                s+=errs[k+1]
            
            if i==j: #The diagonal is just the error, depending on the type
                if types[i]==1:
                    C[i,j] = sig1*sig1*(s+1)
                    #C[i,j] = sig1*sig1+s
                else:
                    #C[i,j] = sig2*sig2*(s+1)
                    C[i,j] = sig2*sig2 +s*0
            else:
                if types[i]==1 and types[j]==1:
                    C[i,j] = sig1*sig1*s
                    C[j,i] = sig1*sig1*s
                    #C[i,j] = s
                    #C[j,i] = s
                elif types[i]==0 and types[j]==0:
                    #C[i,j] = sig2*sig2*s
                    #C[j,i] = sig2*sig2*s
                    C[i,j] = s*0
                    C[j,i] = s*0

                else:
                    #C[i,j] = sig1*sig2*s
                    #C[j,i] = sig1*sig2*s
                    C[i,j] = s*0
                    C[j,i] = s*0
    Cr = numpy.mat(C)
    return Cr"""


#Returns the covariance matrix for a vector of types (which will be either true or rounded types)
def decay_cov_mat(types, t):
    
    om_m = t[0]  
    om_l = t[1]
    H0 = t[2]
    b = t[3]
    sig1 = t[4]
    sig2 = t[5]
    x = t[6]
    y = t[7]
    z = t[8]   
    
    N=len(types)
    C = numpy.zeros([N,N])
    types=numpy.array(types)
    
    #This matrix is the absolute value of the difference between indices
    d=numpy.ones([N, N])*numpy.array(range(N))
   
    dists=abs(d-d.T)
    dists[range(N), range(N)]=1
    B=numpy.mat(types==1)
    
    
    C[B.T*B]=sig1*sig1*x/dists[B.T*B]
    
    sigs=numpy.ones([N])*sig1*sig1
    sigs[numpy.array(types==0)]=sig2*sig2
    
    C[range(N), range(N)]=sigs
    
    return numpy.mat(C)

"""#BACKUP
#Returns the covariance matrix for a vector of types (which will be either true or rounded types)
def decay_cov_mat(types, t):
    
    om_m = t[0]  
    om_l = t[1]
    H0 = t[2]
    b = t[3]
    sig1 = t[4]
    sig2 = t[5]
    x = t[6]
    y = t[7]
    z = t[8]   
    
    N=len(types)
    C = numpy.zeros([N,N])
    for i in range(N):
        for j in range(i, N):
            if i==j: #The diagonal is just the error, depending on the type
                if types[i]==1:
                    C[i,j] = sig1*sig1
                else:
                    C[i,j] = sig2*sig2
            else:
                dist = factor(i, j)
                if types[i]==1 and types[j]==1:
                    C[i,j] = x*sig1*sig1/dist
                    C[j,i] = x*sig1*sig1/dist
                elif types[i]==0 and types[j]==0:
                    C[i,j] = y*sig2*sig2/dist
                    C[j,i] = y*sig2*sig2/dist
                else:
                    C[i,j] = z*sig1*sig2/dist
                    C[j,i] = z*sig1*sig2/dist
    Cr = numpy.mat(C)
    return Cr"""
    

#Returns the covariance matrix for a vector of types (which will be either true or rounded types)
#This is an exponential covariance matrix which is supposed to always be positive definite
def exp_cov_mat(types, t):
    om_m = t[0]  
    om_l = t[1]
    H0 = t[2]
    b = t[3]
    sig1 = t[4]
    sig2 = t[5]
    x = t[6]
    y = t[7]
    z = t[8]
    
    N=len(types)
    C = numpy.zeros([N,N])
    for i in range(N):
        for j in range(i, N):
            if i==j: #The diagonal is just the error, depending on the type
                if types[i]==1:
                    C[i,j] = sig1*sig1
                else:
                    C[i,j] = sig2*sig2
            else:
                dist = factor(i, j)
                if types[i]==1 and types[j]==1:
                    C11=sig1*sig1/numpy.exp(dist/x)
                    C[i,j] = C11
                    C[j,i] = C11
                elif types[i]==0 and types[j]==0:
                    C00=0
                    C[i,j] = C00
                    C[j,i] = C00
                else:
                    C01=0
                    C[i,j] = C01
                    C[j,i] = C01
    Cr = numpy.mat(C)
    return Cr
#One pure population of 1a's
#This function takes the number of data points to create, the standard deviations
#of the 1a mean and 
#standard deviation of redshifts
#It produces a matrix of the distance modulus, (assuming LCDM), redshift and probabilities.
def sn_1a(N,sig1, z_min, z_max):
    #First we spline the distance modulus function, cos it takes to long to integrate
    z_spl = numpy.linspace(z_min, z_max,100)
    y_spl = save_dist(z_spl)

    spline = interpolate.splrep(z_spl,y_spl)
    

    #Make the redshifts
    red = []
    for i in range(N):
        #This is going to give absurd values if z is negative, but might produce a bias if we throw them away
        #z = numpy.random.randn()*red_sig + red_mu
        #For now I'm artificially using a uniform distribution
        z = numpy.random.random()*(z_max-z_min)+z_min
        if z<0:
            print "Negative z: ",z
        red.append(z)

    mod=[]  #Distance moduli
    
    for i in range(N):
        #print i

        #Find the mean of the distance modulus from the spline
        m = interpolate.splev(red[i],spline)
        mod.append(numpy.random.randn()*sig1+m)
    #print numpy.column_stack((red,mod,prob,types))
    prob = numpy.ones(len(mod))
    types = numpy.ones(len(mod))
    return numpy.column_stack((red,mod, prob, types))

    
#Since it takes so long to do these integrals, we just compute a few values,
#save them to file and then spline
def save_dist(z):
    y = []
    for i in range(len(z)):
        y.append(dist_mod(z[i]))
    numpy.savetxt('distances.txt',numpy.column_stack((z,y)))
    return y

    
#Given a redshift, using flat LCDM parameters to compute the distance modulus
def dist_mod(z, *args):
    if len(args)!=0:
        om_m=args[0]
        om_l=args[1]
        H0=args[2]
    else:
        om_m=0.272
        om_l=1-om_m
        H0=70.4
    
    y,err = integrate.quad(dist,0.0,z, (om_m, om_l, H0))
    #Test case for curvature
    om_k = 1.0 - om_m - om_l
    if om_k==0:
        d = c*(1.0+z)/H0*y
    elif om_k<0:
        d = c*(1.0+z)/H0/numpy.sqrt(-om_k)*numpy.sin(numpy.sqrt(-om_k)*y)
    else:
        d = c*(1.0+z)/H0/numpy.sqrt(om_k)*numpy.sinh(numpy.sqrt(om_k)*y)
    return 5.0*numpy.log10(d) + 25.0

#Function to integrate
def dist(z, om_m, om_l, H0):
    return (om_m*(1+z)**3.0+om_l+(1- om_m-om_l)*(1+z)**2.0)**(-0.5)


#Plot some data, colour coded
def plot_sn(data, uncor):
    z=data[:,0]
    y=data[:,1]
    prob=data[:,2]
    types=data[:,3]
    
    z_th = numpy.linspace(min(z), max(z), 50)
    th = save_dist(z_th)
    
    if uncor:
        pylab.errorbar(z[types==1],y[types==1],yerr=sig1, linestyle='none', marker='.', color='c')
        pylab.errorbar(z[types==0],y[types==0],yerr=sig2, linestyle='none', marker='^', color='y')
    else:
        pylab.errorbar(z[types==1],y[types==1],yerr=sig1, linestyle='none', marker='.')
        pylab.errorbar(z[types==0],y[types==0],yerr=sig2, linestyle='none', marker='^', color='#3bf940')
    
    #Subtract off the theory to get the standard deviations
    th_2 = numpy.array(th)+b
    
    mu_Ia=[]
    mu_nIa=[]
    
    spl=interpolate.splrep(z_th, th)
    for i in range(len(z)):
        if types[i]==1:
            mu_Ia.append(y[i] - interpolate.splev(z[i], spl))
        else:
            mu_nIa.append(y[i] - interpolate.splev(z[i], spl)+b)
    
    print numpy.std(mu_Ia), numpy.std(mu_nIa)
    
    pylab.plot(z_th, th, 'r')
    #Plot the current biased best fit line
    y = []
    for i in range(len(z_th)):
        y.append(dist_mod(z_th[i], 0.1, 0.2, 72))
    #pylab.plot(z_th, y, 'm', lw=2)
    pylab.plot(z_th, th_2, 'r--')
    #pylab.title('Mock SN data', fontsize=24)
    pylab.subplots_adjust(top=0.95, bottom=0.1, left=0.1, right=0.95)
    pylab.xlabel(r'$z$',fontsize=24)
    pylab.ylabel(r'$\mu$',fontsize=24)
    pylab.ylim([33, 53])
    pylab.show()

#Plot some data, colour coded
def plot_sn_1a(data):
    z=data[:,0]
    y=data[:,1]
    
    pylab.errorbar(z,y, yerr=sig1, linestyle='none', marker='.')
    pylab.title('Mock SN data')
    pylab.xlabel(r'z',fontsize=16)
    pylab.ylabel(r'$\mu$',fontsize=16)
    pylab.show()


def metadata(root):
    #Save the metadata
    fl = open(root+'data_parameters.txt', 'w')
    fl.write('N = %d \n' %(N))
    fl.write('sig1 = %0.2f \n' %(sig1))
    fl.write('sig2 = %0.2f \n' %(sig2))
    fl.write('b = %0.2f \n' %(b))
    if TYPE=='decay':
        fl.write('x = %0.2f \n' %(x))
        fl.write('y = %0.2f \n' %(y))
        fl.write('z = %0.2f \n' %(z))
    elif TYPE=='block':
        fl.write('s1 = %0.2f \n' %(s1))
        fl.write('s2 = %0.2f \n' %(s2))
        fl.write('s3 = %0.2f \n' %(s3))
        fl.write('s4 = %0.2f \n' %(s4))
        fl.write('s5 = %0.2f \n' %(s5))
    else:
        fl.write('s1 = %0.2f \n' %(S1))
        fl.write('s2 = %0.2f \n' %(S2))
        fl.write('s3 = %0.2f \n' %(S3))
        fl.write('s4 = %0.2f \n' %(S4))
        fl.write('s5 = %0.2f \n' %(S5))
    fl.write('eps = %0.2f \n' %(eps))
    fl.close()
    
def make(root, name):
    #data = sn_uncor(N,sig1,sig2,b,z_min,z_max,eps)
    #numpy.savetxt('SN uncor 10/SN_uncor.txt',data)
    
    #S=1.0

    
    #data = sn_cor(N,sig1,sig2,b, S, S, S, S, S, z_min,z_max,eps)
    if TYPE=='decay':
        data = sn_cor(N,sig1,sig2,b, x, y, z,-1, -1,  z_min,z_max,eps)
    elif TYPE=='wed':
        data = sn_cor(N,sig1,sig2,b, S1, S2, S3, S4, S5, z_min,z_max,eps)
    elif TYPE=='block':
        data = sn_cor(N,sig1,sig2,b, s1,s2,s3,s4,s5, z_min,z_max,eps)
    numpy.savetxt(root+name ,data)
    metadata(root)
    #numpy.savetxt('SN_cor_example.txt',data)
    #data = numpy.loadtxt('SN_cor_example.txt')
    """pylab.figure()
    pylab.hist(data[:, 2], 20)
    pylab.figure()"""
    #plot_sn(data, False)
    
def make_uncor(root):
    #data = sn_uncor(N,sig1,sig2,b,z_min,z_max,eps)
    #numpy.savetxt('SN uncor 10/SN_uncor.txt',data)
    data = sn_uncor(N,sig1,sig2,b, z_min,z_max,eps, False)
    numpy.savetxt(root+'SN_uncor.txt',data)
    metadata(root)
    #data = numpy.loadtxt('SN cor 10/SN_cor.txt')
    #pylab.figure()
    #pylab.hist(data[:, 2], 20)
    #pylab.figure()
    #plot_sn(data)
    
def make_1a():
    data = sn_1a(N,sig1,  z_min,z_max)
    numpy.savetxt('example/SN_1a_example.txt',data)
    #data = numpy.loadtxt('SN cor 10/SN_cor.txt')
    plot_sn_1a(data)
################################ MAIN #########################################
N = 1000
sig1 = 0.1
#sig1 = 0.01
sig2 = 1.5
b = 2.0

z_min = 0.01
z_max = 2.0
eps = 0.1

x = 0.7
y = 0.
z = 0.

s1=0.02
s2=0.04
s3=0.06
s4=0.08
s5=0.1

S1=S2=S3=S4=S5=1.5 #Formally 0.01
#S1=0.0

t0 = [om_m,om_l, H0, b, sig1, sig2, x, y, z] 

#x=0.05
#y=0.03
#z=0.005

#plot_sn(numpy.loadtxt('/media/DATA/Project/BEAMS_Res/SN_cor_marg_200/SN_cor.txt'))
#data = sn_cor(1000,sig1,sig2,b, 0, 0, 0, z_min,z_max,eps)

make_uncor('./')
d=numpy.loadtxt('SN_uncor.txt')
#d=d[d[:, 2]>0.9, :]
plot_sn(d, True)

if 'make' in sys.argv:
    #make('p_BEAMS_test/','SN_cor4.txt')
    #root='/media/DATA/Project/BEAMS_Res/'
    #root='/media/Elements/Backup/BEAMS_Res/'
    #root='/sdcard/Project/BEAMS/'
    #root='/USBdisk1/Backup/BEAMS_Res/'
    root='./'
    if TYPE=='wed':
        #make(root+'all_kim/','SN_cor.txt')
        make(root,'SN_cor_test.txt')
    elif TYPE=='block':
        #make(root+'all_block/','SN_cor.txt')
        make(root,'SN_cor_test.txt')
    elif TYPE=='decay':
        make(root+'all_decay/','SN_cor.txt')
    #make_uncor(root+'All_uncor/')
    #make('SN_cor_x=%0.1f/' %(x),'SN_cor.txt')
    #print
    #make_uncor('SN_uncor_marg_10000/')
   # make_1a()
   
elif 'plot' in sys.argv:
    #plot_sn(numpy.loadtxt('/media/My Passport/Backup/BEAMS_Res/All_decay/SN_cor.txt'))
    data1=numpy.loadtxt('big_sim2/data0/SN_cor.txt')
    plot_sn(data1, False)
   
    #data = sn_uncor(N,sig1,sig2,b, z_min,z_max,eps, False)
    #plot_sn(data, False)
    pylab.show()
    #plot_sn(numpy.loadtxt('/media/My Passport/Backup/BEAMS_Res/all_kim/SN_cor.txt'))
    #plot_sn(sn_uncor(N,sig1,sig2,b, z_min,z_max,eps))
elif 'lots' in sys.argv:
    X=[0.0, 0.1,0.2, 0.3, 0.4, 0.5, 0.6]
    for x in X:
        root='SN_cor_x=%0.1f/' %(x)
        name='SN_cor.txt'
        data = sn_cor(N,sig1,sig2,b, x, y, z, z_min,z_max,eps)
        numpy.savetxt(root+name ,data)
        metadata(root)

elif 'testy' in sys.argv:
    min=-0.2
    max=0.5
    n=10
    X=numpy.linspace(min, max, n)
    Y=numpy.linspace(min, max, n)
    Z=numpy.linspace(min, max, n)
    #X=[100000.0]
    
    d = numpy.loadtxt('SN_cor_marg_200_x2/SN_cor.txt')
    types = d[:, 3]
    #types = [0, 0, 1, 1, 0]
    C0= cov_mat(types, t0)
    print C0
    i=1
    for x in X:
        #for y in Y:
        #    for z in Z:
        #print x, y, z
        print x
        t = t0[:]
        t[6]=x
        t[7]=0.
        t[8]=0
        C = cov_mat(types, t)
        #print t
        #print C
        try:
            Q= numpy.linalg.cholesky(C)
        except numpy.linalg.linalg.LinAlgError:
            print 'not positive definite'
        #pylab.subplot(1, 3, i)
        i+=1
        C=numpy.array(C)
        #pylab.imshow(C)
    #C0=numpy.array(C0)
    
    
    #pylab.imshow(C0)
    #pylab.subplot(1, 2, 2)
    
    #pylab.colorbar()
    #pylab.show()
    
    
    
    
    
    
    
    
    
    
elif 'cov' in sys.argv:
    TYPE='wed'
    S=1.0
    
    #root1='/media/My Passport/Backup/BEAMS_Res/'
    #root1='big_sim_200_more_Ias_more_cor/data0/'
    root1='big_sim2/data0/'
    
    if TYPE=='decay':
        t0 = [om_m,om_l, H0, b, sig1, sig2, x, y, z] 
        types = numpy.loadtxt(root1+'SN_cor.txt')[:, 3]
        C = numpy.array(decay_cov_mat(types, t0))
    
    elif TYPE=='block':
        t0=[om_m,om_l, H0, b, sig1, sig2, s1, s2, s3, s4,  s5] 
        data = numpy.loadtxt(root1+'SN_cor.txt')
        types=data[:, 3]
        C = numpy.array(block_cov_mat(types, t0, data))
    
    else:
        t0=[om_m,om_l, H0, b, sig1, sig2, S1, S2, S3, S4, S5] 
        types = numpy.loadtxt(root1+'SN_cor.txt')[:, 3]
        C = numpy.array(wed_cov_mat(types, t0))    
    
    for i in range(len(C[:, 0])):
        C[i, i]=0
    #Q = numpy.linalg.cholesky(C)
    #pylab.imshow(C, norm=LogNorm())
    #fig = pylab.figure(figsize=(8, 6))
    pylab.imshow(C, interpolation='bicubic', cmap='gist_heat_r')
    #pylab.subplots_adjust(left=-0.2, right=1.0)
    pylab.colorbar()
    
    
    pylab.show()

elif 'big_sim' in sys.argv:
    #root1='/media/Elements/Backup/BEAMS_Res/big_sim/'
    root1='block_sim/'

    os.system('mkdir '+root1)
    metadata(root1)
    for i in range(10):
        root=root1+'data'+(str)(i)+'/'
        os.system('mkdir '+root)
        os.system('cp SNe_Binned_Covariance.txt '+root)
        make(root, 'SN_cor.txt')
        #if i==9:
         # plot_sn(numpy.loadtxt(root1+'SN_cor_9.txt'))
        #pylab.show()
    
          
elif 'big_sim_uncor' in sys.argv:
    root1='big_sim_chi2/'
    #metadata(root1)
    os.system('mkdir '+root1)
    for i in range(20):
        root=root1+'data'+(str)(i)+'/'
        os.system('mkdir '+root)
        make_uncor(root)
        
elif 'big_sim_uncor_Ia' in sys.argv:
    root1='big_sim_uncor_Ia/'
    for i in range(10):
        root=root1+'data'+(str)(i)+'/'
        os.system('mkdir '+root)
        data = sn_uncor(N,sig1,sig2,b, z_min,z_max,eps, True)
        numpy.savetxt(root+'SN_uncor.txt',data)
        metadata(root)
        if i==9:
            plot_sn_1a(data)

elif 'test_block' in sys.argv:
    c=numpy.loadtxt('SNe_Binned_Covariance.txt')
    
    #d=numpy.loadtxt('SN_cor_test.txt')
    root='big_sim_200_more_Ias_more_cor/data0/'
    d=numpy.loadtxt(root+'SN_cor.txt')
    types=d[:, 3]
    a=time.time()
    cb=block_cov_mat(types, t0, d)
    b=time.time()
    print 'first', b-a, 's'
    
    a=time.time()
    cb=block_cov_mat(types, t0, d)
    b=time.time()
    print 'second', b-a, 's'
    
    pylab.figure()
    pylab.imshow(c)
    pylab.colorbar()
    
    N=len(cb[:, 0])
    cb[range(N), range(N)]=0
    pylab.figure()
    pylab.imshow(cb, interpolation='bicubic', cmap='hot')
    pylab.colorbar()
    
    pylab.show()
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
