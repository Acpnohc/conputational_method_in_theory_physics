# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 01:12:42 2021

@author: iluvatar
"""

import numpy as np
import matplotlib.pyplot as plt

def plt_pdf(p,low,high,dt,h):
    
    global size
    global x
    global y
    
    size = int((high-low)/h)
    
    x = np.linspace(0,1,size)
    
    x = low + x*(high-low)
    
    y = np.zeros(size)
    
    for i in range(size):
        
        y[i] = ( p.size - (p < x[i] - dt/2).sum() - (p > x[i] + dt/2).sum() )/p.size/dt
    
    y[0] = y[0]*2
    
    return [x,y]

p = []

a = np.random.uniform(size=10**5)*(10**5)

a = a.tolist()

a.sort()

for i in range(len(a)-1):
    
    p.append(a[i+1]-a[i])

p = np.array(p)

s = plt_pdf(p,min(p),max(p),0.01,0.01)

plt.plot(s[0],s[1],label='numerical')

plt.ylabel('P(v)')

plt.xlabel('v')

plt.show()