from scipy.special import gamma
import numpy as np

def f_Eqn_6(g,d,b):
    '''
    Returns the value of Eqn 6 in Tian's 2023 paper, for
    input values for gamma (g), delta (d) and beta (b).
    '''
    ep = g+d-2
    A1 = 3-d
    A2 = (ep-2*b)*(3-ep)
    B1 = gamma(0.5*(ep-1))/gamma(ep/2)
    B2 = b*gamma(0.5*(ep+1))/gamma(0.5*(ep+2))
    C1 = gamma(g/2)*gamma(d/2)
    C2 = gamma(0.5*(g-1))*gamma(0.5*(d-1))
    f = (A1/A2)*(B1-B2)*(C1/C2)
    return f

