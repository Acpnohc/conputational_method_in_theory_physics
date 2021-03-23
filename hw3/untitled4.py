#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 21:51:07 2021

@author: JChonpca_Huang
"""

import numpy as np
import matplotlib.pyplot as plt

def plt_pdf(p,low,high,dt,h):
    
    size = int((high-low)/h)
    
    x = np.linspace(0,1,size)
    
    x = low + x*(high-low)
    
    y = np.zeros(size)
    
    for i in range(size):
        
        y[i] = ( p.size - (p < x[i] - dt/2).sum() - (p > x[i] + dt/2).sum() )/p.size/dt
    
    return [x,y]

p = np.random.uniform(size=100000)

p = (p+1)**2

s = plt_pdf(p,min(p),max(p),0.1,0.01)

plt.plot(s[0],s[1],label='numerical')

xx = np.linspace(1,4,10000)

yy = 1/(2*np.sqrt(xx))


plt.plot(xx,yy,label='theoretical')

plt.ylabel('P(v)')

plt.xlabel('v')

plt.legend()

plt.show()

plt.show()