import sys
CLUS=False #Are we running this on the cluster
if CLUS==False:
    sys.path.insert(0,'/home/knights/numpy-1.6.1/build/lib.linux-x86_64-2.6/')
    import numpy
    #print numpy.version.version
    
else:
    import numpy
print numpy.version.version
import random
from scipy import interpolate
from scipy import integrate
import SN_data
import decimal
import time
import sys

#import pylab

#Constants
c = 2.99792e5 #In km/s
decimal.getcontext().prec=7 #How many decimal places we'll carry around

#Use these variables for the priors and for choosing random variables before starting the chain
P_om = [-0.2, 1.2] #I've put these back to the ones which worked on the 30/07/2012.  20/11 - changed to 1.2 to avoid negative dist
P_ol = [-0.2, 1.2]
P_H0 = [10, 130]
P_x = [0.3, 0.9]
mux=0.08
sigx=0.05

P_sig1 = [numpy.log(0.05), numpy.log(0.3)]
P_sig2 = [numpy.log(0.5), numpy.log(4.5)]
#P_sig1=[0.05, 0.15]
#P_sig2=[1.0, 2.0]
P_b = [1.0,3.0]
P_S = [-0.5,2.0]
#P_S = [0.0005, 0.0015]
#P_S = [-0.01, 0.03]
P_Sb = [0.0,0.12]

#Wedding cake
#WED=True
#I think the same system arguments that get passed to the mcmc, come here

chi2_cov=[] #The covariance matrix for the correlated chi^2 likelihood
chi2_coeff=0

#Flat prior for the parameter set t (old) and u (new)
#Changing this to test!
def prior(t_old, t):
    
    #We now check to see if this combination of parameters causes the equation which you
    #take the square root of in the distance modulus to go negative:
    om_m=t[0]
    om_l=t[1]
    z_max = 2.0
    if om_m*(1+z_max)**3.0+om_l+(1-om_m-om_l)*(1+z_max)**2.0<0:
        return 0
    elif t[0]<P_om[0] or t[0]>P_om[1]:
        return 0
    elif t[1]<P_ol[0] or t[1]>P_ol[1]:
        return 0
    elif t[2]<P_H0[0] or t[2]>P_H0[1]:
        return 0
    elif t[3]<P_b[0] or t[3]>P_b[1]:
        return 0
    elif t[4]<P_sig1[0] or t[4]>P_sig1[1]:
        return 0
    elif t[5]<P_sig2[0] or t[5]>P_sig2[1]:
        return 0
        
        
    if 'WED' in sys.argv:
      for i in range(6,11):
        if t[i]<P_S[0] or t[i]>P_S[1]:
          return 0
          
    elif 'BLOCK' in sys.argv:
      for i in range(6,11):
        if t[i]<P_Sb[0] or t[i]>P_Sb[1]:
          return 0
          
    elif 'DECAY' in sys.argv:
      if t[6]<P_x[0] or t[6]>P_x[1]:
        return 0
#      elif t[7]<P_x[0] or t[7]>P_x[1]:
#        return 0
#      elif t[8]<P_x[0] or t[8]>P_x[1]:
#        return 0        
    return 1
    
    #elif t[7]<0.4 or t[7]>0.6:
    #    return 0
    #elif t[8]<0.1 or t[8]>0.3:
    #    return 0    
#    else:
#        if t[6]==t_old[6]: #We're not varying x
#            Px=1
#        else:
#            Px = gauss(t[6], mux, sigx)/gauss(t_old[6], mux, sigx)
#            #print 'old', t_old[6], 'new', t[6]
#        return Px

     

def prior_renee(t_old, t):
    
    #We now check to see if this combination of parameters causes the equation which you
    #take the square root of in the distance modulus to go negative:
    om_m=t[0]
    om_l=t[1]
    z_max = 1.5
    if om_m*(1+z_max)**3.0+om_l+(1-om_m-om_l)*(1+z_max)**2.0<0:
        return 0
    elif t[0]<P_om[0] or t[0]>P_om[1]:
        return 0
    elif t[1]<P_ol[0] or t[1]>P_ol[1]:
        return 0
    elif t[2]<P_H0[0] or t[2]>P_H0[1]:
        return 0
    elif t[6]<P_sig1[0] or t[6]>P_sig1[1]:
        return 0
    elif t[7]<P_sig2[0] or t[7]>P_sig2[1]:
        return 0
    else:
        return 1

#Baby function to produce gaussian
def gauss(x, mu, sig):
    return 1.0/numpy.sqrt(2.0*numpy.pi*sig*sig)*numpy.exp(-(x-mu)*(x-mu)/2.0/sig/sig)

#BEAMS uncorrelated likelihood
#Data must contain the objects and their probabilities
def beams_uncor_like_DEC(t,data):
    om_m = t[0]  
    om_l = t[1]
    H0 = t[2]
    b = t[3]
    sig1 = numpy.exp(t[4])
    sig2 = numpy.exp(t[5])
    #sig1=t[4]
    #sig2=t[5]
    
    #if prior(t)==0:
    #    return decimal.Decimal('0')
    
    #Spline the distance modulus function over the redshifts spanned by the data
    z_spl = numpy.linspace(min(data[:, 0]), max(data[:, 0]),50)
    y_spl = dist(z_spl, om_m,om_l,  H0)
    spline = interpolate.splrep(z_spl, y_spl)
    
    max_num=300.0
    num=float(len(data[:, 0]))
    #Sum over the loglikelihood
    like=decimal.Decimal('1')
    for i in range(len(data[:,0])):
        #Predicted mu
        mu = interpolate.splev(data[i, 0], spline)
        #Likelihood for each population
        coeff = 1.0/numpy.sqrt(2*numpy.pi)/sig1
        x=(-(mu-data[i,1])*(mu-data[i,1])/sig1/sig1/2.0)  
        La = toDecimal(coeff, x)
        coeff = 1.0/numpy.sqrt(2*numpy.pi)/sig2
        x = (-(mu+b-data[i,1])*(mu+b-data[i,1])/sig2/sig2/2.0)
        Lb = toDecimal(coeff, x)
        
        pi = decimal.Decimal((str)(data[i, 2]))
        like=like*(La*pi + Lb*(decimal.Decimal('1')-pi))
        #print total
    return like
    
#BEAMS uncorrelated likelihood
#Data must contain the objects and their probabilities
def beams_uncor_like(t,data):
    om_m = t[0]  
    om_l = t[1]
    H0 = t[2]
    b = t[3]
    sig1 = numpy.exp(t[4])
    sig2 = numpy.exp(t[5])
    #sig1=t[4]
    #sig2=t[5]
    
    #if prior(t)==0:
    #    return decimal.Decimal('0')
    
    #Spline the distance modulus function over the redshifts spanned by the data

    z_spl = numpy.linspace(min(data[:, 0]), max(data[:, 0]),50)
    y_spl = dist(z_spl, om_m,om_l,  H0)
    spline = interpolate.splrep(z_spl, y_spl)

    
    max_num=300.0
    num=float(len(data[:, 0]))
    #Sum over the loglikelihood
    like=1.0
    for i in range(len(data[:,0])):
        #Predicted mu
        mu = interpolate.splev(data[i, 0], spline)
        #Likelihood for each population
        coeff = 1.0/numpy.sqrt(2*numpy.pi)/sig1
        x=(-(mu-data[i,1])*(mu-data[i,1])/sig1/sig1/2.0)  
        La = coeff*numpy.exp(x)
        coeff = 1.0/numpy.sqrt(2*numpy.pi)/sig2
        x = (-(mu+b-data[i,1])*(mu+b-data[i,1])/sig2/sig2/2.0)
        Lb = coeff*numpy.exp(x)
        
        pi = data[i, 2]
        like=like*(La*pi + Lb*(1-pi))
        #print total
    if like==0:
        like=1e-300
    return like
    
#Function to evaluate the distance modulus for a vector of redshifts
def dist(z, om_m, om_l, H0):
    #For now, I'm enforcing flatness
    #om_l = 1-om_m
    
    mu=[]
    for k in range(len(z)):
        y,err, msg = integrate.quad(func,0.0,z[k],args=(om_m, om_l), full_output=1)
        #Test case for curvature
        om_k = 1.0 - om_m - om_l
        if om_k==0:
            d = c*(1.0+z[k])/H0*y
        elif om_k<0:
            d = c*(1.0+z[k])/H0/numpy.sqrt(-om_k)*numpy.sin(numpy.sqrt(-om_k)*y)
        else:
            d = c*(1.0+z[k])/H0/numpy.sqrt(om_k)*numpy.sinh(numpy.sqrt(om_k)*y)
        mu.append(5.0*numpy.log10(d) + 25.0)
    return mu
    
#Function for dist to integrate
def func(z, om_m, om_l):
    """print om_m
    print om_l
    print 1-om_m-om_l
    print"""
    if om_m*(1+z)**3.0+om_l+(1-om_m-om_l)*(1+z)**2.0<0:
        print 'negative'
        print om_m
        print om_l
        print z
    return (om_m*(1+z)**3.0+om_l+(1-om_m-om_l)*(1+z)**2.0)**(-0.5)

    
#Perturbed BEAMS likelihood
def perturbed_uncor_like(t,data):
    om_m = t[0]  
    om_l = t[1]
    H0 = t[2]
    b = t[3]
    sig1 = t[4]
    sig2 = t[5]
    
    #Ok since I'm sick of dealing with rubbish when the chain jumps outside the prior, I'll put
    #in a check here
    #if prior(t)==0:
     #   return decimal.Decimal('0')

    prob = numpy.array(data[:,2])
    Pr = prob.round() #Rounded probabilities
    #To try and reduce confusion, a type "a" object is tau=1 and a type "b" object is type 0
    Tr = numpy.copy(Pr) #Rounded types
    
    eps = prob-Pr #The small deviations from the rounded probabilities
    
    #Spline the distance modulus function over the redshifts spanned by the data
    z_spl = numpy.linspace(min(data[:, 0]), max(data[:, 0]),50)
    y_spl = dist(z_spl, om_m, om_l, H0)
    spline = interpolate.splrep(z_spl, y_spl)
    
    likeR = beams_like(Tr,t, spline,data) #The zeroth order term of the likelihood 
     
    max_num=300.0
    num=float(len(data[:, 0]))
    total = likeR
    
    #Now we run through the data and calculate the first order term
    for i in range(len(data[:,0])):

        #mu = interpolate.splev(data[i, 0], spline) #Distance modulus
        
        #like += eps[i] * (beams_like(tau1,t,data)-beams_like(tau2,t,data)) * P
        D = data[i, :].reshape([1, len(data[i, :])]) #Deals with numpy expecting 2d arrays etc
        likeA = beams_like([1], t,spline, D) #The probability if the i'th point was a 1a
        likeB = beams_like([0], t,spline, D) #The probability if the i'th point was a non-1a
        """if likeA<10.0**(-max_num/num):
                likeA=10.0**(-max_num/num)
        if likeB<10.0**(-max_num/num):
                likeB=10.0**(-max_num/num)"""
        #We take the rounded likelihood, divide out the i'th term and then multiply by the new term
        if Tr[i]==1: #It's already a 1a
            like1=likeR
            like2 = likeR/likeA*likeB
        else: #It's already a non-1a
            like1 = likeR/likeB*likeA
            like2=likeR
        
        total += decimal.Decimal(str(eps[i]))*(like1-like2)
        #print total
    #print 'zero: ', likeR
    #print 'first: ', total - likeR
    #print
    return total
    
#Perturbed, correlated BEAMS likelihood
def perturbed_cor_like(t,data):
    SECOND=False #Use the second order term
    ZERO = False #Use only the zeroth order term
    
    om_m = t[0]  
    om_l = t[1]
    H0 = t[2]
    b = t[3]
    sig1 = t[4]
    sig2 = t[5]
    x = t[6]
    y = t[7]
    z = t[8]
    
    #Ok since I'm sick of dealing with rubbish when the chain jumps outside the prior, I'll put
    #in a check here
    #if prior(t)==0:
     #   return decimal.Decimal('0')

    N=len(data[:,0])
    
    #Spline the data once with these parameters and pass the spline to the likelihood function
    z_spl = numpy.linspace(min(data[:, 0]), max(data[:, 0]),50)
    y_spl = dist(z_spl, om_m, om_l, H0)
    spline = interpolate.splrep(z_spl, y_spl)
    
    """pylab.plot(data[:, 0], data[:, 1], '.')
    pylab.plot(data[:, 0], interpolate.splev(data[:, 0], spline))
    pylab.show()"""

    P = numpy.array(data[:,2]) #Probabilities
    Pr = P.round() #Rounded probabilities
    #To try and reduce confusion, a type "a" object is tau=1 and a type "b" object is type 0
    Tr = numpy.array(P.round()) #Rounded types
    
    #We can now construct the covariance matrix
    Cr=SN_data.cov_mat(Tr, t)

    likeR = beams_cor_like(Tr,t,Cr,data, spline) #The zeroth order term of the likelihood 
    total = likeR
    #print 'zero: ', likeR
    
    if ZERO:
        return likeR
    #print 'zero: ', likeR
    eps = P-Pr #The small deviations from the rounded probabilities
    
    #print 'first: '
    first = decimal.Decimal('0') #First order term
    #Now we run through the data and calculate the first order term
    for i in range(N):
        if Tr[i]==1: #It's already a type 1
            C2 = recompute(Cr, Tr, t, i, 0)
            T2 = numpy.copy(Tr)
            T2[i]=0
            like1 = likeR
            #print 'like1 ', like1
            like2 = beams_cor_like(T2,t, C2,data, spline)
            #print 'like2 ', like2
            
        else:
            C1 = recompute(Cr, Tr, t, i, 1)
            T1 = numpy.copy(Tr)
            T1[i] = 1
            like1 = beams_cor_like(T1,t, C1,data, spline)
            like2 = likeR           
        #print i, ': ', decimal.Decimal(str(eps[i]))*(like1-like2)
        #print '1a like: ', like1
        #print 'non-1a like: ', like2
        #print
        first += decimal.Decimal(str(eps[i]))*(like1-like2)
        #print first
        #print total
    
    total = likeR+first
    
    if SECOND:
        print 'second:'
        second=decimal.Decimal('0')
        for i in range(N):
            for j in range(N):
                if i != j:
                    if Tr[i]==1: #It's already a type 1
                        if Tr[j]==1: #like11=likeR
                            like11 = likeR
                            C10 = recompute(Cr, Tr, t, j, 0)
                            T10 = numpy.copy(Tr)
                            T10[j]=0
                            like10 = beams_cor_like(T10,t, C10,data, spline)
                            
                            C01 = recompute(Cr, Tr, t, i, 0)
                            T01 = numpy.copy(Tr)
                            T01[i]=0
                            C00 = recompute(C01, T01,t, j, 0)
                            T00 = numpy.copy(T01)
                            T00[j]=0
                            like01 = beams_cor_like(T01, t, C01, data, spline)
                            like00 = beams_cor_like(T00, t, C00, data, spline)
                        else: #like10 = likeR
                            like10 = likeR
                            C11 = recompute(Cr, Tr, t, j, 1)
                            T11 = numpy.copy(Tr)
                            T11[j]=1
                            like11 = beams_cor_like(T11, t, C11, data, spline)
                            
                            C00 = recompute(Cr, Tr, t, i, 0)
                            T00 = numpy.copy(Tr)
                            T00[i]=0
                            C01 = recompute(C00, Tr, t, j, 1)
                            T01 = numpy.copy(T00)
                            T01[j]=1
                            like00 = beams_cor_like(T00, t, C00, data, spline)
                            like01 = beams_cor_like(T01, t, C01, data, spline)
                    
                    else:
                        if Tr[j]==1: #like01=likeR
                            like01 = likeR
                            C00 = recompute(Cr, Tr, t, j, 0)
                            T00 = numpy.copy(Tr)
                            T00[j]=0
                            like00 = beams_cor_like(T00, t, C00, data, spline)
                            
                            C11 = recompute(Cr, Tr, t, i, 1)
                            T11 = numpy.copy(Tr)
                            T11[i]=1
                            C10 = recompute(C11, T11, t, j, 0)
                            T10 = numpy.copy(T11)
                            T10[j]=0
                            like11 = beams_cor_like(T11, t, C11, data, spline)
                            like10 = beams_cor_like(T10, t, C10, data, spline)
                            
                        else: #like00=likeR
                            like00 = likeR
                            C01 = recompute(Cr, Tr, t, j, 1)
                            T01 = numpy.copy(Tr)
                            T01[j]=1
                            like01 = beams_cor_like(T01, t, C01, data, spline)
                            
                            C10 = recompute(Cr, Tr, t, i, 1)
                            T10 = numpy.copy(Tr)
                            T10[i]=1
                            C11 = recompute(C10, T10, t, j, 1)
                            T11 = numpy.copy(T10)
                            T11[j] = 1
                            like10 = beams_cor_like(T10, t, C10, data, spline)
                            like11 = beams_cor_like(T11, t, C11, data, spline)
                    second += decimal.Decimal((str)(eps[i]))*decimal.Decimal((str)(eps[j]))*(like11+like00-like10-like01)
                    #print i, ', ', j, ': ', decimal.Decimal((str)(eps[i]))*decimal.Decimal((str)(eps[j]))*(like11+like00-like10-like01)
        total += decimal.Decimal('0.5')*second
        #print Tr
    #print 'first: ', first
    #print likeR
    #print 'zero order: ', likeR
    #print 'first order: ', first
    #print 'second order: ', second
    return total



#Returns a correlated likelihood, given a "types" vector and a covariance matrix
def beams_cor_like_DEC(tau,t,C,data, spline):   
    b = t[3]

    
    #Construct the theory minus data vector
    th=numpy.zeros(len(tau))
    for i in range(len(data[:, 0])):
        mu = interpolate.splev(data[i, 0], spline)
        #print 'mu ', mu
        if tau[i]==1:
            th[i] = mu
        else:
            th[i] = mu + b
    delta = numpy.mat(th-data[:,1])

    #There's still the problem of C being infinite, even in the decimal class
    #Solved this using slogdet!
    #print numpy.linalg.det(C)
    
    coeff = 1.0/(2*numpy.pi)**(len(data[:,0])/2.0)
    #sign, logdet = numpy.linalg.slogdet(C)
    #det = toDecimal(sign, logdet)
    det = decimal.Decimal((str)(numpy.linalg.det(C)))
    
    #There's a possibility this determinant is negative, this is clearly not a combination of parameters we want
    if det<=decimal.Decimal('0'):
        print 'negative determinant'
        return decimal.Decimal('0')
    coeff = decimal.Decimal(str(coeff))/numpy.sqrt(det)

    x = -0.5*float(delta*C.I*delta.T)
    like = toDecimal(coeff, x)
    
    return like

#Returns a correlated likelihood, given a "types" vector and a covariance matrix
def beams_cor_like(tau,t,C,data, spline):   
    b = t[3]

    #Construct the theory minus data vector
    th=numpy.zeros(len(tau))
    for i in range(len(data[:, 0])):
        mu = interpolate.splev(data[i, 0], spline)
        #print 'mu ', mu
        if tau[i]==1:
            th[i] = mu
        else:
            th[i] = mu + b
    delta = numpy.mat(th-data[:,1])

    #There's still the problem of C being infinite, even in the decimal class
    #Solved this using slogdet!
    #print numpy.linalg.det(C)
    
    coeff = 1.0/(2*numpy.pi)**(len(data[:,0])/2.0)
    #sign, logdet = numpy.linalg.slogdet(C)
    #det = toDecimal(sign, logdet)
    det = numpy.linalg.det(C)
    
    #There's a possibility this determinant is negative, this is clearly not a combination of parameters we want
    if det<=0:
        print 'negative determinant'
        return 0
    coeff = coeff/numpy.sqrt(det)

    x = -0.5*float(delta*C.I*delta.T)
    like = coeff*numpy.exp(x)
    if like==0:
        like=1e-300
    
    return like

#This little function takes a number to be converted to decimal
#Takes two inputs, A and x, such that the number is A*exp(x)
#This is based on {exp(x) = a*10^y} where y is an integer
def toDecimal(A, x):
    #There's a possibility that x will just be rubbish because we've jumped somewhere outside the prior
    if numpy.isinf(x) or numpy.isnan(x):
        return decimal.Decimal('0')
    else:
        y = round(x/numpy.log(10))
        a = numpy.exp(x-numpy.log(10)*y)
        y = int(y)
    
    return decimal.Decimal(str(a)+'e'+str(y))*decimal.Decimal(str(A))



#BEAMS likelihood calculation for a given types (tau) for all the data points, as well as the
#theory. Data must include the probabilities for each object
def beams_like(tau,t,spline, data):
    om_m = t[0]  
    om_l = t[1]
    H0 = t[2]
    b = t[3]
    sig1 = t[4]
    sig2 = t[5]
    #print 'tau ', tau

    like=decimal.Decimal('1')
    for i in range(len(data[:,0])):
        #Predicted mu
        mu = interpolate.splev(data[i, 0], spline)
        #Now construct the likelihood, dependent on the types in tau
        if tau[i]==1:
            coeff=1.0/numpy.sqrt(2*numpy.pi)/sig1
            x = -(mu-data[i,1])*(mu-data[i,1])/sig1/sig1/2.0
            """print 'x ', x
            print mu
            print data[i, 1]
            print"""
            l = toDecimal(coeff, x)
            #print -(mu-data[i,1])*(mu-data[i,1])/sig1/sig1/2.0
        else:
            coeff = 1.0/numpy.sqrt(2*numpy.pi)/sig2
            x = -(mu+b-data[i,1])*(mu+b-data[i,1])/sig2/sig2/2.0
            """print 'x ', x
            print mu
            print data[i, 1]
            print"""
            l = toDecimal(coeff, x)
            #print -(mu+b-data[i,1])*(mu+b-data[i,1])/sig2/sig2/2.0
        #We have to deal with numbers which go below the smallest number python can handle
        like = like*l
        #print like
    #print 'likelihood ', like
    return like 

#Function which, given the i'th index and the type it should be, recomputes the covariance matrix
#with the changed type
def recompute(C_old, T, t, ind, new_type):
    sig1 = t[4]
    sig2 = t[5]
    x = t[6]
    y = t[7]
    z = t[8]
    
    C=C_old.copy()
    #Change the row and column of the index
    for i in range(len(T)):
        if i==ind: #The diagonal is just the error, depending on the type
            if new_type==1:
                C[i,ind] = sig1*sig1
            else:
                C[i,ind] = sig2*sig2
        else:
            dist = SN_data.factor(i, ind) #This is the factor for the covariance matrix we are using
            if T[i]==1 and new_type==1:
                new = x*sig1*sig1/dist
                C[i,ind] = new
                C[ind,i] = new
            elif T[i]==0 and new_type==0:
                new = y*sig2*sig2/dist
                C[i, ind]= new
                C[ind,i] = new
            else:
                new = z*sig1*sig2/dist
                C[i, ind]= new
                C[ind,i] = new
    return C


#Full BEAMS likelihood 
def true_beams(t, spline, data):
    
    #if prior(t)==0:
    #    return decimal.Decimal('0')
    om_m = t[0]  
    om_l = t[1]
    H0 = t[2]
    b = t[3]
    sig1 = t[4]
    sig2 = t[5]
    #x = t[6]
    #y = t[7]
    #z = t[8]
    
    #Spline the data once with these parameters and pass the spline to the likelihood function
    #z_spl = numpy.linspace(min(data[:, 0]), max(data[:, 0]),50)
    #y_spl = dist(z_spl, om_m, om_l, H0)
    #spline = interpolate.splrep(z_spl, y_spl)
    
    total=decimal.Decimal('0')
    N=len(data[:, 0])
    #Now we iterate over all possible types by counting in binary

    for i in range(2**N):
        arr = numpy.array(list(bin(i))[2:], 'int')
        tau = numpy.hstack((numpy.zeros(N-len(arr)),arr))

#        if i==0:
#            tau_old = numpy.ones(len(tau))
#            C_old = numpy.mat(numpy.zeros([len(tau), len(tau)]))
#            #Use the compute_cov function to compute the first cov matrix 
#        C = compute_cov(C_old, tau_old, tau, t)
        C=SN_data.wed_cov_mat(tau, t)
        like = beams_cor_like(tau, t, C, data, spline)
        
        #Find the P(tau)
        P=1.0
        for j in range(N):
            if tau[j]==1:
                P=P*data[j, 2]
            else:
                P=P*(1-data[j, 2])
        P = decimal.Decimal((str)(P))
        
#        tau_old=tau.copy()
#        C_old = C.copy()
        
        total+=like*P
        
    return total
        
#This takes an old covariance matrix and updates by a new tau vector
def compute_cov(C_old, T_old, T_new, t):
    sig1 = t[4]
    sig2 = t[5]
    x = t[6]
    y = t[7]
    z = t[8]
    
    C = C_old.copy()
    #Find the indices we need to change
    indices = numpy.where(numpy.array(T_old)-numpy.array(T_new)!=0)[0]
    
    #Now we go through those indices we need to change and update the covariance matrix
    for ind in indices:
        #Go through this row/ column
        for i in range(len(C[:, 0])):
            if i==ind:
                if T_new[ind]==1:
                    C[ind, ind]=sig1*sig1
                else:
                    C[ind, ind]=sig2*sig2
            else:
                dist = SN_data.factor(i, ind) #This is the factor for the covariance matrix we are using
                if T_new[i]==1 and T_new[ind]==1:
                    new = x*sig1*sig1/dist
                    C[i, ind]=new
                    C[ind,  i]=new
                elif T_new[i]==0 and T_new[ind]==0:
                    new = y*sig2*sig2/dist
                    C[i, ind]=new
                    C[ind,  i]=new
                else:
                    new = z*sig1*sig2/dist
                    C[i, ind]=new
                    C[ind,  i]=new
                        
    return C        




#Apply a naive cut
def chi2(t_in, data):
    t=t_in[:]
    om_m = t[0]  
    om_l = t[1]
    H0 = t[2]
    b = t[3]
    sig1 = t[4]
    sig2 = t[5]
    
   # if prior(t)==0:
    #    return decimal.Decimal('0')
        
    z_spl = numpy.linspace(min(data[:, 0]), max(data[:, 0]),50)
    y_spl = dist(z_spl, om_m, om_l, H0)
    spline = interpolate.splrep(z_spl, y_spl)
    
    if DEC:
        like=decimal.Decimal('1')
    else:
        like=1
    for i in range(len(data[:,0])):
        #Predicted mu
        
        #Apply a cut according to probability
        if data[i, 2]>0.9:
            mu = interpolate.splev(data[i, 0], spline)
            coeff=1.0/numpy.sqrt(2*numpy.pi)/sig1
            x = -(mu-data[i,1])*(mu-data[i,1])/sig1/sig1/2.0
            if DEC:
                l = toDecimal(coeff, x)
            else:
                l=coeff*numpy.exp(x)
            
            like = like*l
    return like 
"""
##TESTING##
N = 2
sig1 = 0.1
sig2 = 2.0
b = 3.0
z_min = 0.01
z_max = 1.5
eps = 0.1

H0 = 70.4 #In km/s/Mpc
om_m = 0.272

x=0.5
y=0.3
z = 0.05

t = [om_m, H0, b, sig1, sig2, x, y, z] 
data = SN_data.sn_cor(N,sig1,sig2,b,x, y, z, z_min,z_max,eps)
print 'data'
print data

P = numpy.array(data[:,2]) #Probabilities
Pr = P.round() #Rounded probabilities
#To try and reduce confusion, a type "a" object is tau=1 and a type "b" object is type 0
Tr = numpy.array(P.round()) #Rounded types

#We can now construct the covariance matrix
C = numpy.zeros([N,N])

for i in range(N):
    for j in range(N):
        if i==j: #The diagonal is just the error, depending on the type
            if Tr[i]==1:
                C[i,j] = sig1*sig1
            else:
                C[i,j] = sig2*sig2
        elif Tr[i]==1 and Tr[j]==1:
            C[i,j] = x*sig1*sig1/abs(i-j)
        elif Tr[i]==0 and Tr[j]==0:
            C[i,j] = y*sig2*sig2/abs(i-j)
        else:
            C[i,j] = z*sig1*sig2/abs(i-j)
Cr = numpy.mat(C)

print 'cov, ', Cr

print 'like ', beams_cor_like(Tr,t,Cr,data)"""











#Apply a naive cut
def chi2_cor(t_in, data):
    global chi2_cov, chi2_coeff
    t=t_in[:]
    om_m = t[0]  
    om_l = t[1]
    H0 = t[2]
    b = t[3]
    sig1 = numpy.exp(t[4])
    sig2 = numpy.exp(t[5])
    t[4]=sig1
    t[5]=sig2
    
    #if len(chi2_cov)==0:
    #For now we just always assuming wedding cake cov mat
    chi2_cov=SN_data.wed_cov_mat([1]*len(data[:, 0]), t)
    det = numpy.linalg.det(chi2_cov)
    if det<1e-300:
        det=1e-300
    chi2_coeff = 1.0/(2*numpy.pi)**(len(data[:,0])/2.0)/numpy.sqrt(det)
    """numpy.set_printoptions(threshold=numpy.nan)
    print
    print chi2_cov
    print 'det', numpy.linalg.det(chi2_cov)

    print chi2_coeff"""
        

    z_spl = numpy.linspace(min(data[:, 0]), max(data[:, 0]),50)
    y_spl = dist(z_spl, om_m, om_l, H0)
    spline = interpolate.splrep(z_spl, y_spl)
    
    mu=interpolate.splev(data[:, 0], spline)
    delta = numpy.mat(mu-data[:,1])

    like = chi2_coeff*numpy.exp(-0.5*float(delta*chi2_cov.I*delta.T))
    if like==0:
        like=1e-300
    
    return like

#This is special p-BEAMS. We expand about the maximum likelihood terms instead
#I don't think this really works.
def special_perturbed_cor_like(t,data):
    ZERO = False #Use only the zeroth order term
    
    om_m = t[0]  
    om_l = t[1]
    H0 = t[2]
    b = t[3]
    sig1 = t[4]
    sig2 = t[5]
    x = t[6]
    y = t[7]
    z = t[8]
    
    #Ok since I'm sick of dealing with rubbish when the chain jumps outside the prior, I'll put
    #in a check here
   # if prior(t)==0:
    #    return decimal.Decimal('0')

    N=len(data[:,0])
    
    #Spline the data once with these parameters and pass the spline to the likelihood function
    z_spl = numpy.linspace(min(data[:, 0]), max(data[:, 0]),50)
    y_spl = dist(z_spl, om_m, om_l, H0)
    spline = interpolate.splrep(z_spl, y_spl)
    
    """pylab.plot(data[:, 0], data[:, 1], '.')
    pylab.plot(data[:, 0], interpolate.splev(data[:, 0], spline))
    pylab.show()"""

    P = numpy.array(data[:,2]) #Probabilities
    Pr = P.round() #Rounded probabilities
    #To try and reduce confusion, a type "a" object is tau=1 and a type "b" object is type 0
    Tr = numpy.array(P.round()) #Rounded types
    
    #Using the fiducial model, we try to find which types destroy the likelihood and reverse them
    tf = SN_data.t0
    Cf = SN_data.cov_mat(Tr, tf)
    like0 = beams_cor_like(Tr,tf,Cf,data, spline)
    #print 'like0 ', like0
    T_true=Tr.copy()
    P_true=P.copy()
    for k in range(N):
        T = Tr.copy()
        T[k] = 1-T[k]
        like = beams_cor_like(T, tf, SN_data.cov_mat(T, t), data, spline)
        #print k
        #print like/like0
        if like/like0>decimal.Decimal('100'):
            
            T_true[k]=T[k]
            P_true[k]=1-P[k]
            #print 'changed ', k
    Tr=T_true.copy()
    #We can now construct the covariance matrix
    Cr=SN_data.cov_mat(Tr, t)

    likeR = beams_cor_like(Tr,t,Cr,data, spline) #The zeroth order term of the likelihood 
    total = likeR
    #print 'zero: ', likeR
    
    if ZERO:
        return likeR
    #print 'zero: ', likeR
    #eps = P-Pr #The small deviations from the rounded probabilities
    eps=P_true-T_true
    
    #print 'first: '
    first = decimal.Decimal('0') #First order term
    #Now we run through the data and calculate the first order term
    for i in range(N):
        if Tr[i]==1: #It's already a type 1
            C2 = recompute(Cr, Tr, t, i, 0)
            T2 = numpy.copy(Tr)
            T2[i]=0
            like1 = likeR
            #print 'like1 ', like1
            like2 = beams_cor_like(T2,t, C2,data, spline)
            #print 'like2 ', like2
            
        else:
            C1 = recompute(Cr, Tr, t, i, 1)
            T1 = numpy.copy(Tr)
            T1[i] = 1
            like1 = beams_cor_like(T1,t, C1,data, spline)
            like2 = likeR           
        #print i, ': ', decimal.Decimal(str(eps[i]))*(like1-like2)
        #print '1a like: ', like1
        #print 'non-1a like: ', like2
        #print
        first += decimal.Decimal(str(eps[i]))*(like1-like2)
        #print first
        #print total
    
    total = likeR+first
    return total



#BEAMS uncorrelated likelihood
#Data must contain the objects and their probabilities
#BEAMS likelihood using the decimal class, may be slower
def beams_renee_dec(t,data):
    om_m = t[0]  
    om_l = t[1]
    H0 = t[2]
    a0=t[3]
    a1=t[4]
    a2=t[5]
    sig1 = numpy.exp(t[6]) #we take steps in log sigma
    sig2 = numpy.exp(t[7])
    
    #if prior(t)==0:
    #    return decimal.Decimal('0')
    
    #Spline the distance modulus function over the redshifts spanned by the data
    z_spl = numpy.linspace(min(data[:, 0]), max(data[:, 0]),50)
    y_spl = dist(z_spl, om_m,om_l,  H0)
    spline = interpolate.splrep(z_spl, y_spl)

    num=float(len(data[:, 0]))
    #Sum over the loglikelihood
    like=decimal.Decimal('1')
    for i in range(len(data[:,0])):
        #Predicted mu
        mu = interpolate.splev(data[i, 0], spline)
        #Likelihood for each population
        coeff = 1.0/numpy.sqrt(2*numpy.pi)/sig1
        x=(-(mu-data[i,1])*(mu-data[i,1])/sig1/sig1/2.0)  
        La = toDecimal(coeff, x)
        coeff = 1.0/numpy.sqrt(2*numpy.pi)/sig2
        x = (-(mu+a0+a1*data[i, 0] + a2*data[i, 0]*data[i, 0]-data[i,1])*(mu+a0+a1*data[i, 0] + a2*data[i, 0]*data[i, 0]-data[i,1])/sig2/sig2/2.0)
        Lb = toDecimal(coeff, x)
        
        pi = decimal.Decimal((str)(data[i, 2]))
        like=like*(La*pi + Lb*(decimal.Decimal('1')-pi))
        #print total
    return like

#BEAMS uncorrelated likelihood
#Data must contain the objects and their probabilities
def beams_renee(t,data):
    om_m = t[0]  
    om_l = t[1]
    H0 = t[2]
    a0=t[3]
    a1=t[4]
    a2=t[5]
    sig1 = numpy.exp(t[6]) #we take steps in log sigma
    sig2 = numpy.exp(t[7])
    
    #if prior(t)==0:
    #    return decimal.Decimal('0')
    
    #Spline the distance modulus function over the redshifts spanned by the data
    z_spl = numpy.linspace(min(data[:, 0]), max(data[:, 0]),50)
    y_spl = dist(z_spl, om_m,om_l,  H0)
    spline = interpolate.splrep(z_spl, y_spl)

    num=float(len(data[:, 0]))
    #Sum over the loglikelihood
    #like=decimal.Decimal('1')
    loglike=0
    for i in range(len(data[:,0])):
        #Predicted mu
        mu = interpolate.splev(data[i, 0], spline)
        #Likelihood for each population
        coeff = 1.0/numpy.sqrt(2*numpy.pi)/sig1
        x=(-(mu-data[i,1])*(mu-data[i,1])/sig1/sig1/2.0)  
        #La = toDecimal(coeff, x)
        La = coeff*numpy.exp(x)
        coeff = 1.0/numpy.sqrt(2*numpy.pi)/sig2
        x = (-(mu+a0+a1*data[i, 0] + a2*data[i, 0]*data[i, 0]-data[i,1])*(mu+a0+a1*data[i, 0] + a2*data[i, 0]*data[i, 0]-data[i,1])/sig2/sig2/2.0)
        #Lb = toDecimal(coeff, x)
        Lb = coeff*numpy.exp(x)
        
        #pi = decimal.Decimal((str)(data[i, 2]))
        pi = data[i, 2]
        loglike+=numpy.log((La*pi + Lb*(1.0-pi)))
        #print total
    return loglike



DEC=False

if 'testing' in sys.argv:
    """t0=[0.27,0.73,70,1.5,1.0,-3.0,numpy.log(0.3),numpy.log(2.0)]
    #root='/USBdisk1/Backup/BEAMS_Res/Gaussian/'
    root='Gaussian/'
    data_name = root+'renee.txt'
    data = numpy.loadtxt(data_name)
    a=time.time()
    print beams_renee(t0,data)
    b=time.time()
    print b-a,'s'"""
    H0 = 70.4 #In km/s/Mpc
    om_m = 0.272
    om_l = 1-om_m
    sig1 = 0.1
    sig2 = 1.5
    b = 2.0
    S1=S2=S3=S4=S5=1.5 #Formally 0.01
    S1=0.0
    t0 = [om_m,om_l, H0, b, sig1, sig2, S1, S2, S3, S4, S5] 
    #t0 = [om_m,om_l, H0, b, sig1, sig2, 0.7, 0, 0] 

    root='big_sim2/data0/'
    data=numpy.loadtxt(root+'SN_cor.txt')
    
    t=t0[:]
    for i in range(len(data[:, 3])):
        t.append(data[i, 3])
    
    
    
    types=data[:, 3]
    types=types[types==1]
    data=data[types==1]
    z_spl = numpy.linspace(min(data[:, 0]), max(data[:, 0]),50)
    y_spl = dist(z_spl, om_m, om_l, H0)
    spline = interpolate.splrep(z_spl, y_spl)
    #types=numpy.array([1, 1, 1, 1, 1, 1, 1, 1, 1])
    a=time.time()
    #numpy.set_printoptions(threshold=numpy.nan)
    C1=SN_data.wed_cov_mat(types, t0)
    print numpy.linalg.det(C1)
    print C1
    print beams_cor_like(data[:, 3],t0,C1,data, spline)
    b=time.time()
    print b-a,'s'
    
    """a=time.time()
    C2=SN_data.decay_cov_mat(types, t0)
    b=time.time()
    print b-a,'s'

    #numpy.set_printoptions(threshold=numpy.nan)
    print sum(numpy.array(sum(C1-C2))[0])"""
    
    """ a=time.time()
    print beams_uncor_like(t0,data)
    b=time.time()
    print b-a,'s'
    
    a=time.time()
    #print beams_cor_like(data[:, 3],t0,C,data, spline)
    print marg_like(t, data)
    b=time.time()
    print b-a,'s'"""
