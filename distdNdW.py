__all__ = ["dNdW"]

import numpy as np
import math

def dNdW(data, p): #WEAK + STRONG
    return (p[3]*np.exp(-data/p[2])/p[2] + (p[1])*np.exp(-data/p[0])/p[0])

#p[0] = Wstar_weak
#p[1] = Nstar_weak
#p[2] = Wstar_str
#p[3] = Nstar_str
