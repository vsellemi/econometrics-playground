#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  9 10:35:18 2021

@author: victorsellemi
"""

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

def plot_ts(date, series, labels = ["NA"], style = "solid", col = "darkblue", 
            outfile = "NA", lw = 1, marker = '.', ms = 5):
    
    """
    Function to plot time series object
    
    Input:
        date = date vector or list with length T
        series = Txk matrix of series to plot
        labels = label for each series
        
    """
    
    # plot settings
    sns.set(context = "paper", style = "white", font = "serif", font_scale = 1)
    
    if len(series.shape)>1:
        _,k = series.shape
    else:
        k=1
        
    
    if k > 1:
        fig, ax = plt.subplots()
        plt.gca().set_prop_cycle(plt.cycler('color', cm.plasma(np.arange(0,k+1)/k)))
        ax.yaxis.grid()
        for i in range(k):
            ax.plot(date,series[:,i], label = labels[i], linewidth = lw, linestyle = style, marker = marker, markersize=ms)
        ax.legend()
        fig.tight_layout()
        if outfile != "NA":
            plt.savefig(outfile, dpi = 300)
        plt.show()
    
    else:
        
        fig, ax = plt.subplots()
        ax.yaxis.grid()
        ax.plot(date,series, linewidth = lw, linestyle = style, color = col, marker = marker, markersize=ms)
        fig.tight_layout()
        if outfile != "NA":
            plt.savefig(outfile, dpi = 300)
        plt.show()
        
        
    
    
    
    
