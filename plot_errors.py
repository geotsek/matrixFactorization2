#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 18:40:42 2023

@author: georgios
"""

import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import rc
from matplotlib import rcParams


if __name__ == '__main__':

    dataFolder = '/Users/georgios/MyStuff/MyData/geneDoidMatrixFactorData/'

    #in_edges_csv = dataFolder + 'binetGeneDoidEdges_small.csv'
    in_edges_csv = dataFolder + 'binetGeneDoidEdges.csv'

    out_data = dataFolder + 'data_ndims.txt'
    #out_data = dataFolder + 'data_nepochs.txt'

    fig_png_file = dataFolder + 'plot' + 'Errors' + '_ndims.png'
    #fig_pdf_file = dataFolder + 'plot' + 'Errors' + '_ndims.pdf'
    #fig_png_file = dataFolder + 'plot' + 'Errors' + '_nepochs.png'
    #fig_pdf_file = dataFolder + 'plot' + 'Errors' + '_nepochs.pdf'
    
    data = np.loadtxt(out_data)
    data = data.T
    font = {'size' :14}
    matplotlib.rc('font', **font)
    #plt.figure(figsize=(6,6))
    fig, ax = plt.subplots(2,1, sharey=True, gridspec_kw={'hspace': 0.3}, figsize=(6,8))
    #rcParams.update({'figure.autolayout': True})


    fac1=100

    #ax[0].plot(data[0],data[4],'mo:')
    #ax[0].plot(data[0],data[5],'ro:')
    #ax[0].plot(data[0],data[6],'go:')
    #ax[0].plot(data[0],data[7],'bo:')
    ax[0].semilogx(data[0],data[4],'mo:')
    ax[0].semilogx(data[0],data[5],'ro:')
    #ax[0].semilogx(data[0],data[6],'go:')
    #ax[0].semilogx(data[0],data[7],'bo:')
    #ax[0].loglog(xs1,ys1,'ko')
    #ax[0].set_xticks([5000,15000])
    #ax[1].set_yticks([1e-4,1e-3,1e-2,1e-1,1e-0])
    ax[0].set_xlabel('n_dims')
    #ax[0].set_xlabel('n_epochs')
    ax[0].set_ylabel('error')


    #ax[1].plot(data[0],data[6],'go:')
    #ax[1].plot(data[0],data[7],'bo:')
    ax[1].semilogx(data[0],data[6],'go:')
    ax[1].semilogx(data[0],data[7],'bo:')
    #ax[1].plot(data[0],data[4]/data[1],'mo:')
    #ax[1].plot(data[0],data[5]/data[1],'ro:')
    #ax[1].plot(data[0],data[6]/data[1],'go:')
    #ax[1].plot(data[0],data[7]/data[1],'bo:')
    #ax[1].semilogx(data[0],data[4]/data[1],'mo:')
    #ax[1].semilogx(data[0],data[5]/data[1],'ro:')
    #ax[1].semilogx(data[0],data[6]/data[1],'go:')
    #ax[1].semilogx(data[0],data[7]/data[1],'bo:')
    #ax[1].loglog(xs2,ys2,'ko')
    #ax[0].set_xticks([5000,15000])
    #ax[1].set_yticks([1e-4,1e-3,1e-2,1e-1,1e-0])
    ax[1].set_xlabel('n_dims')
    #ax[1].set_xlabel('n_epochs')
    #ax[1].set_ylabel('error relative')
    ax[1].set_ylabel('error rounded')

    ax[0].set_title('funk-svd ' + ' n_epochs=' + str(100) ,size=12)
    #ax[0].set_title('funk-svd ' + ' n_dims=' + str(8) ,size=12)

    #plt.savefig(fig_pdf_file) # save as pdf
    plt.savefig(fig_png_file,dpi=400) # save as png
    plt.show()
