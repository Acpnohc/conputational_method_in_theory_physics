#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 20:01:01 2021

@author: JChonpca_Huang
"""

import numpy as np
import matplotlib.pyplot as plt
import random

def f(x):
    
    if -1 < x < 1:
        
        return np.e**(-1*(np.abs(x)**(5/3)))
    
    else:
        
        return 0
    

def plt_pdf(p,low,high,dt,h):
    
    size = int((high-low)/h)
    
    x = np.linspace(0,1,size)
    
    x = low + x*(high-low)
    
    y = np.zeros(size)
    
    for i in range(size):
        
        y[i] = ( p.size - (p < x[i] - dt/2).sum() - (p > x[i] + dt/2).sum() )/p.size/dt
    
    return [x,y]


p = []

p.append(-1 + 2*random.random())

for i in range(1,1000):
    
    new = p[-1] -1 + 2*random.random()
    
    if f(new)/f(p[i-1]) > 1:
        
        p.append(new)
    
    else:
        
        if random.random() < f(new)/f(p[i-1]):
            
            p.append(new)
            
            
        else:
            
            p.append(p[-1])


p = np.array(p)

s = plt_pdf(p,min(p),max(p),0.1,0.01)

plt.plot(s[0],s[1],label='numerical')

xx = np.linspace(-1,1,10000)

yy = 0.69*np.e**(-1*(np.abs(xx)**(5/3)))

yy[0] = 0

yy[-1] = 0

plt.plot(xx,yy,label='theoretical')

plt.ylabel('P(v)')

plt.xlabel('v')

plt.legend()

plt.show()