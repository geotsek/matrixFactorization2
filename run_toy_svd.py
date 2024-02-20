#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 14:35:35 2024

@author: georgios
"""

import sys
#sys.path.append('FastRP-master')

import csv

import numpy as np
import scipy as sp
from scipy.linalg import svd




if __name__ == '__main__':

    dataFolder = '/Users/georgios/MyStuff/MyData/geneDoidMatrixFactorData/'


    in_edges_csv = dataFolder + 'binetGeneDoidEdges_small.csv'
    #in_edges_csv = dataFolder + 'binetGeneDoidEdges.csv'

    reader = csv.reader(open(in_edges_csv, "rb"), delimiter=",")


    # Singular-value decomposition
    # define a matrix
    A = np.array([[1, 2], [3, 4], [5, 6]])
    print('A=',A)
    # SVD
    U, s, VT = svd(A)
    
    print('U=',U)
    print('s=',s)
    print('VT=',VT)
    
    
    n = np.shape(A)
    # create m x n Sigma matrix
    Sigma = np.zeros((n[0], n[1]))
    # populate Sigma with n x n diagonal matrix
    Sigma[:n[1], :n[1]] = np.diag(s)
    print('Sigma=',Sigma)

    # reconstruct matrix
    B = np.matmul(U,np.matmul(Sigma,VT))
    print('B=',B)