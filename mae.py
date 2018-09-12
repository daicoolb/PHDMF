'''
Created on July 29, 2017
@author: Beili
'''

from __future__ import print_function

import sys

import numpy as np

if sys.version_info.major == 2:
    range = xrange


def eval_MAE(R, U, V, TS):
    num_user = U.shape[0]
    sub_mae = np.zeros(num_user)
    TS_count = 0
    for i in range(num_user):
        idx_item = TS[i]
        if len(idx_item) == 0:
            continue
        TS_count = TS_count + len(idx_item)
        approx_R_i = U[i].dot(V[idx_item].T)  # approx_R[i, idx_item]
        R_i = R[i]
        sub_mae[i] = np.abs(approx_R_i - R_i).sum()
    mae = sub_mae.sum() / TS_count
    return mae

from data_manager import Data_Factory
#import numpy as np
data_factory=Data_Factory()
data_path='/home/daicoolb/CopeData/test_0723_0.8_100k'
#R,D_all,S=data_factory.load(data_path)
#train_user=data_factory.read_rating(data_path+'/train_user.dat')
test_user=data_factory.read_rating(data_path+'/test_user.dat')
U=np.loadtxt(data_path+'/Result/U.dat')
V=np.loadtxt(data_path+'/Result/V.dat')
print("MAE: %.5f \n " % eval_MAE(test_user[1],U,V,test_user[0]))
