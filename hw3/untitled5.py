#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 22:01:13 2021

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

p = []


for i in range(100000):
    
    p.append(np.random.uniform(size=10).sum())

p = np.array(p)

s = plt_pdf(p,min(p),max(p),0.1,0.01)

plt.plot(s[0],s[1],label='numerical')


plt.ylabel('P(v)')

plt.xlabel('v')

#plt.legend()

plt.show()

plt.show()