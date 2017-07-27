#!/usr/bin/python

#coding=utf-8

from evaluation import eval_precision,eval_recall
from data_manager import Data_Factory
import numpy as np

data_factory=Data_Factory()
data_path='/home/daicoolb/CopeData/test_0723_0.8_100k'
train_user=data_factory.read_rating(data_path+'/train_user.dat')
test_user=data_factory.read_rating(data_path+'/test_user.dat')
valid_user=data_factory.read_rating(data_path+'/valid_user.dat')
U=np.loadtxt(data_path+'/Result/U.dat')
V=np.loadtxt(data_path+'/Result/V.dat')
print "Precision @5: %.5f \n" % eval_precision(U,V,test_user,train_user,valid_user,50)
print "Recall @5:% .5f \n" % eval_recall(U,V,test_user,train_user,valid_user,50)
