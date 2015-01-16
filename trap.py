#!/Users/gmsardane/Library/Enthought/Canopy_64bit/User/bin/python
from scipy.integrate import trapz
import numpy as np

n = 32
m = 50

x = np.linspace(1., 3., n)
y = np.linspace(0., 4., m)
z = np.zeros(m)

print len(y),len(z)

i = 0

for i in x:


"""
for i in x:
 for j in y
  z(i,j) = (i+3*j)**3

trapz(dist, xx)

f = trapz(y,trapz(x,z))

"""
#ans =
#     6.466052394922165e+03
