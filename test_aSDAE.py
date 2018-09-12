#!/usr/bin/python
#conding=utf-8

from __future__ import print_function

from text_analysis.aSDAE import aSDAE_module
import numpy as np
import os
import sys

if sys.version_info.major == 2:
    import cPickle as pickl

else:
    import pickle as pickl


#num_item=1000
#num_user=100
user_feature=200
output_dimension_module=30
first_dimension_module=100

S=pickl.load(open("/home/daicoolb/Desktop/CopeData/test_0717/side.all","rb"))
R=pickl.load(open("/home/daicoolb/Desktop/CopeData/test_0717/ratings.all","rb"))

#np.random.seed(112)
#SDAE=np.random.uniform(size=(num_user,num_item))
#np.savetxt('./test/sdae.dat',SDAE)
#np.random.seed(112)
#Train_user=np.random.uniform(size=(num_user,user_feature))
#np.savetxt('./test/train_user.dat',Train_user)

aSDAE_out=aSDAE_module(first_dimension_module,output_dimension_module,R.shape[1],S.shape[1])

feature_1=np.random.uniform(size=(R.shape[0],output_dimension_module))
#np.savetxt('./test/feature_input.dat',feature_1)
history=aSDAE_out.train(R.toarray(),S.toarray(),feature_1,112)

module_output=aSDAE_out.get_middle_layer(R.toarray(),S.toarray())

print("done")
np.savetxt("./test/sdae_output.dat",module_output[0])
np.savetxt("./test/train_user_output.dat",module_output[1])
np.savetxt("./test/feature_output.dat",module_output[2])
