#!/usr/bin/python

#coding=utf-8

from __future__ import print_function

import sys

from evaluation import eval_precision,eval_recall
from data_manager import Data_Factory
import numpy as np


data_factory=Data_Factory()
data_path='/home/daicoolb/test_0724_0.8_amazon'
train_user=data_factory.read_rating(data_path+'/train_user.dat')
test_user=data_factory.read_rating(data_path+'/test_user.dat')
valid_user=data_factory.read_rating(data_path+'/valid_user.dat')
U=np.loadtxt(data_path+'/Result/U.dat')
V=np.loadtxt(data_path+'/Result/V.dat')
print("Precision @5: %.5f \n" % eval_precision(U,V,test_user,train_user,valid_user,5))
print("Recall @5:% .5f \n" % eval_recall(U,V,test_user,train_user,valid_user,5))

print("Precision @10: %.5f \n" % eval_precision(U,V,test_user,train_user,valid_user,10))
print("Recall @10:% .5f \n" % eval_recall(U,V,test_user,train_user,valid_user,10))

print("Precision @15: %.5f \n" % eval_precision(U,V,test_user,train_user,valid_user,15))
print("Recall @15:% .5f \n" % eval_recall(U,V,test_user,train_user,valid_user,15))

print("Precision @30: %.5f \n" % eval_precision(U,V,test_user,train_user,valid_user,30))
print("Recall @30:% .5f \n" % eval_recall(U,V,test_user,train_user,valid_user,30))

print("Precision @50: %.5f \n" % eval_precision(U,V,test_user,train_user,valid_user,50))
print("Recall @50:% .5f \n" % eval_recall(U,V,test_user,train_user,valid_user,50))
