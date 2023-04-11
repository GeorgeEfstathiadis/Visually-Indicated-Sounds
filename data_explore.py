# open .mat and .pk files
import scipy.io
import numpy as np

data = scipy.io.loadmat('vis-sfs/vis-data/2015-02-16-17-27-53_sf.mat')
data['sfs'].shape # (time, 45, 42)

# open corresponding .pk file
import pickle
with open('vis-sfs/vis-data/2015-02-16-17-27-53_sf.pk', 'rb') as f:
    data2 = pickle.load(f, encoding='latin1')
np.sum(data2 != data['sfs']) # 0 # same data

