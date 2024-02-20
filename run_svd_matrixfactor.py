#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 14:35:35 2024

@author: georgios
"""

import sys
#sys.path.append('FastRP-master')

import csv

import array

import numpy as np
import scipy as sp
from scipy.linalg import svd

from scipy.sparse import csr_matrix
from scipy.sparse import linalg as sla                                                           

#from collections import Counter
from collections import OrderedDict


def from_csv_to_csr(fp):
    
    edge_list = list()
    #load the csv file
    with open(in_edges_csv, encoding='utf-8-sig', newline='') as f1:
        reader1 = csv.reader(f1)
        edge_list = [tuple(row) for row in reader1]
        
    n_edges = len(edge_list)
    print('n_edges=',n_edges)
        
    nodeA_list = [a for (a,b) in edge_list]
    nodeB_list = [b for (a,b) in edge_list]
    
    nodeA_list2 = list(OrderedDict.fromkeys(nodeA_list))
    nodeB_list2 = list(OrderedDict.fromkeys(nodeB_list))
    n_A = len(nodeA_list2)
    n_B = len(nodeB_list2)
    print('n_A=',n_A,'  n_B=',n_B)

    
    edge_list_id = [(nodeA_list2.index(a),nodeB_list2.index(b)) for (a,b) in edge_list]
    ii = [i for (i,j) in edge_list_id]
    jj = [j for (i,j) in edge_list_id]
    
    n_edges_id = len(edge_list_id)
    print('n_edges_id=',n_edges_id)
    ones = list(np.ones(n_edges))
    
    A = csr_matrix( (ones, (ii,jj)) )
    
    return A

def index_of_sum_until_threshold(x, threshold):
    
    xsum = np.sum(x)
    y = np.cumsum(x)     
    ix = np.where(y<threshold*xsum)[0]
    #print('ix=',ix)
    idx = ix[-1]
    #print('idx=',idx)
    
    return idx
    

if __name__ == '__main__':

    dataFolder = '/Users/georgios/MyStuff/MyData/geneDoidMatrixFactorData/'

    in_edges_csv = dataFolder + 'binetGeneDoidEdges_small.csv'
    #in_edges_csv = dataFolder + 'binetGeneDoidEdges.csv'

    tiny = 1.e-10
    threshold = 0.95

    #A = from_csv_to_csr(in_edges_csv)
    #A = from_csv_to_csr(in_edges_csv).todense()
    
    
    # Singular-value decomposition

    # define a matrix
    A = np.array([[1, 2, 0 , 0], [3, 0, 0, 4], [0, 5, 0, 6]])   
    print('A=',A)
    # SVD
    U, s, VT = svd(A)
    #SVD spare linalg 
    #U, s, VT = sla.svds(A)
    s[s<tiny]=0.0
    
    idx = index_of_sum_until_threshold(s,threshold)
    s[idx+1:] = 0
    
    print('U=',U)
    print('s=',s)
    print('VT=',VT)
    n_diag = s.shape[0]
    
    n = A.shape
    n_max = max(n)
    n_min = min(n)
    # create m x n Sigma matrix
    Sigma = np.zeros((n[0], n[1]))
    # populate Sigma with n x n diagonal matrix
    Sigma[:n_diag, :n_diag] = np.diag(s)
    print('Sigma=',Sigma)

    # reconstruct matrix
    B = np.matmul(U,np.matmul(Sigma,VT))
    print('B=',B)
    
    B[B<tiny]=0
    Asum = np.sum(A)
    ABabsdiff = np.sum(np.abs(A-B))
    ABabsdiffrelative = ABabsdiff/Asum    
    