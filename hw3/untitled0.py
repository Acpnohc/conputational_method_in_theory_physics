#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 15:20:24 2021

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
    
p = np.random.uniform(size=10000)

p = np.sqrt(-2*np.log(1-p))

s = plt_pdf(p,min(p),max(p),0.1,0.01)

yy = s[0]*np.e**(-(1/2)*s[0]**2)

plt.plot(s[0],s[1],label='numerical')

plt.plot(s[0],yy,label='theoretical')

plt.ylabel('P(v)')

plt.xlabel('v')

plt.legend()

plt.show()