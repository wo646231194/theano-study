import numpy as np

print 0 * np.nan
print np.nan == np.nan
print np.inf > np.nan
print np.nan - np.nan
print 0.3 == 3 * 0.1

Z = np.diag(1+np.arange(4),k=-1)
print(Z)