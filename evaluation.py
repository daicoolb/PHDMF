#!/usr/bin/python
#coding=utf-8

from __future__ import print_function


import sys

import numpy as np

if sys.version_info.major == 2:
    range = xrange


def eval_precision(U,V,TS,TN,VN,k):
    num_user=U.shape[0]
    TS_count=0
    result_sum=0
    num_item=V.shape[0]
    print(num_user)
    print(num_item)
    for i in range(num_user):
        idx_item=TN[0][i]
        idx_item_ts=TS[0][i]
        if len(idx_item_ts)==0 and len(VN[0][i])==0:
            continue
        TS_count+=1
        item_candidate=set([ temp_i for temp_i in range(num_item)]) - set(idx_item)
        R_i=U[i].dot(V[list(item_candidate)].T)
        temp_R_i=list(zip(R_i,list(item_candidate)))
        temp_R_i.sort(key=lambda x:x[0],reverse=True)
        predicated=[x[1] for x in temp_R_i]
        act_set=set(TS[0][i]) | set(VN[0][i])
        pre_set=set(predicated[:k])
        result=len(act_set & pre_set)/float(k)
        result_sum+=result
    print(result_sum)
    print(TS_count)
    precision_k=result_sum/TS_count
    return precision_k

def eval_recall(U,V,TS,TN,VN,k):
    num_user=U.shape[0]
    num_item=V.shape[0]
    TS_count=0
    result_sum=0
    for i in range(num_user):
        idx_item=TN[0][i]
        idx_item_ts=TS[0][i]
        if len(idx_item_ts)==0 and len(VN[0][i])==0:
            continue
        TS_count+=1
        item_candidate=set([ temp_i for temp_i in range(num_item)]) - set(idx_item)
        R_i=U[i].dot(V[list(item_candidate)].T)
        temp_R_i=list(zip(R_i,list(item_candidate)))
        temp_R_i.sort(key=lambda x:x[0],reverse = True)
        predicated=[x[1] for x in temp_R_i]
        act_set=set(TS[0][i]) | set(VN[0][i])
        pre_set=set(predicated[:k])
        result=len(act_set & pre_set)/float(len(act_set))
        result_sum+=result
    print(result_sum)
    print(TS_count)
    recall_k=result_sum/TS_count
    return recall_k
