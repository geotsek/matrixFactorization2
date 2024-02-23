#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 12:15:30 2024

@author: georgios
"""

import sys
sys.path.append('funk-svd-master')

from funk_svd import SVD

import csv
import pandas as pd
import numpy as np
#import scipy as sp
import random

from collections import OrderedDict

from sklearn.metrics import mean_absolute_error, mean_squared_error


def from_csv_to_ratings(fp):
    
    edge_list = list()
    #load the csv file
    with open(fp, encoding='utf-8-sig', newline='') as f1:
        reader1 = csv.reader(f1)
        edge_list = [tuple(row) for row in reader1]
        
    n_edges = len(edge_list)
    print('n_edges=',n_edges)
    #print('edge_list[0:5]=',edge_list[0:5])

        
    nodeA_list = [a for (a,b) in edge_list]
    nodeB_list = [b for (a,b) in edge_list]
    
    nodeA_list2 = list(OrderedDict.fromkeys(nodeA_list))
    nodeB_list2 = list(OrderedDict.fromkeys(nodeB_list))
    n_A = len(nodeA_list2)
    n_B = len(nodeB_list2)
    print('n_A=',n_A,'  n_B=',n_B)

    
    list_id = [(nodeA_list2.index(a),nodeB_list2.index(b), 1.0) for (a,b) in edge_list]
    
    return list_id


def from_csv_to_ratings_full(fp):
    
    edge_list = list()
    #load the csv file
    with open(fp, encoding='utf-8-sig', newline='') as f1:
        reader1 = csv.reader(f1)
        edge_list = [tuple(row) for row in reader1]
        
    n_ones = len(edge_list)
    print('n_ones=',n_ones)
        
    nodeA_list = [a for (a,b) in edge_list]
    nodeB_list = [b for (a,b) in edge_list]
    
    nodeA_list2 = list(OrderedDict.fromkeys(nodeA_list))
    nodeB_list2 = list(OrderedDict.fromkeys(nodeB_list))
    n_A = len(nodeA_list2)
    n_B = len(nodeB_list2)
    print('n_A=',n_A,'  n_B=',n_B)
    n_all = n_A * n_B
    print('n_all=',n_all)

    list_all = [(i,j) for i in range(n_A) for j in range(n_B)]
    #print('list_all=',list_all)
    list_ones = [(nodeA_list2.index(a),nodeB_list2.index(b)) for (a,b) in edge_list]
    set_ones = set(list_ones)
    #print('list_ones=',list_ones)
    list_zeros = [x for x in list_all if x not in set_ones]
    #print('list_zeros=',list_zeros)
     
    list_zeros_rand = random.sample(list_zeros,n_ones)
    n_zeros_rand = len(list_zeros_rand)
    print('n_zeros_rand=',n_zeros_rand)
    list_id = [(i,j,1.0) for (i,j) in list_ones] + [(i,j,0.0) for (i,j) in list_zeros_rand]
    n_id = len(list_id)
    print('n_id=',n_id)

    list_id = sorted(list_id, key=lambda element: (element[0], element[1]))
    #print('list_id=',list_id)

    return list_id



if __name__ == '__main__':

    dataFolder = '/Users/georgios/MyStuff/MyData/geneDoidMatrixFactorData/'

    #in_edges_csv = dataFolder + 'binetGeneDoidEdges_small.csv'
    in_edges_csv = dataFolder + 'binetGeneDoidEdges.csv'

    out_data = dataFolder + 'data.txt'

    tiny = 1.e-10
    threshold = 0.95

    #A = from_csv_to_ratings(in_edges_csv)
    A = from_csv_to_ratings_full(in_edges_csv)

    df = pd.DataFrame(A, columns=['u_id', 'i_id', 'rating'])
    df.sort_values(['u_id', 'i_id'], ascending=[True, True])

    train = df.sample(frac=0.8, random_state=7)
    val = df.drop(train.index.tolist()).sample(frac=0.5, random_state=8)
    test = df.drop(train.index.tolist()).drop(val.index.tolist())

    #epochs
    params = [pow(2,i) for i in range(0,10)]
    #dims
    #params = [pow(2,i) for i in range(0,10)]
    
    data_all = []
    for ip in params:
    
        data = []
        n_epochs = ip
        #n_dims = ip
        n_dims = 8
        
        svd = SVD(lr=0.001, reg=0.005, n_epochs=n_epochs, n_factors=n_dims, early_stopping=False,
          shuffle=False, min_rating=0, max_rating=1)

        svd.fit(X=train, X_val=val)

        pred = svd.predict(test)
    
        print('n_dims=',n_dims)

        mae = mean_absolute_error(test['rating'], pred)
        rmse = np.sqrt(mean_squared_error(test['rating'], pred))
        print(f'Test MAE: {mae:.2f}')
        print(f'Test RMSE= {rmse:.2f}')
    
        pred_round = [ round(elem, 0) for elem in pred ]
        mae_round = mean_absolute_error(test['rating'], pred_round)
        rmse_round = np.sqrt(mean_squared_error(test['rating'], pred_round))
        print(f'Test MAE_round: {mae_round:.2f}')
        print(f'Test RMSE_round= {rmse_round:.2f}')

        sum_test = sum(test['rating'])
        sum_pred = sum(pred)
        sum_pred_round = sum(pred_round)
        print(f'Test MAE: {mae:.2f}')
        
        #data.append(n_dims)
        data.append(ip)
        data.append(sum_test)
        data.append(sum_pred)
        data.append(sum_pred_round)
        data.append(mae)
        data.append(rmse)
        data.append(mae_round)
        data.append(rmse_round)
        
        data_all.append(data)
    
    data_ar = np.array(data_all)
    np.savetxt(out_data, data_ar)
    
    
