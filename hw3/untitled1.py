#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 18:07:59 2021

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

p1 = np.random.uniform(size=10000)
p2 = np.random.uniform(size=10000)

p1 = -1 + 2*p1
p11 = np.e**(-1*(np.abs(p1)**(5/3)))

p2 = 5*p2

p = []

for i in range(10000):
    
    if p2[i] < p11[i]:
        
        p.append(p1[i])

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