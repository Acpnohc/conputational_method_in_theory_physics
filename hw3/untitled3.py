#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 20:47:35 2021

@author: JChonpca_Huang
"""

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

xm1 = []
ym1 = []

xm2 = []
ym2 = []


for k in range(100,10000,100):
    
    pp = 0
    
    p = []

    p.append(-1 + 2*random.random())

    for i in range(1,k):
        
        new = -1 + 2*random.random()
        
        if np.e**(-1*(np.abs(new)**(5/3)))/np.e**(-1*(np.abs(p[i-1])**(5/3))) > 1:
            
            p.append(new)
        
        else:
            
            pp += 1
            
            if random.random() < np.e**(-1*(np.abs(new)**(5/3)))/np.e**(-1*(np.abs(p[i-1])**(5/3))):
                
                p.append(new)
                
                
            
            else:
                
                p.append(p[-1])
                
                
    
    p = np.array(p)

    s = plt_pdf(p,min(p),max(p),0.2,0.01)
    
    yy = 0.69*np.e**(-1*(np.abs(s[0])**(5/3)))
    
    xm1.append(pp+k)
    
    ym1.append(((s[1]-yy)**2).sum())
    
    
    
    p1 = np.random.uniform(size=k)
    p2 = np.random.uniform(size=k)
    
    p1 = -1 + 2*p1
    p11 = np.e**(-1*(np.abs(p1)**(5/3)))
    
    p2 = 5*p2
    
    p = []
    
    for i in range(k):
        
        if p2[i] < p11[i]:
            
            p.append(p1[i])
    
    p = np.array(p)

    s = plt_pdf(p,min(p),max(p),0.2,0.01)
    
    yy = 0.69*np.e**(-1*(np.abs(s[0])**(5/3)))
    
    xm2.append(2*k)
    
    ym2.append(((s[1]-yy)**2).sum())
    
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
    

plt.plot(xm1,ym1,label='MC')

plt.plot(xm2,ym2,label='AR')

plt.ylabel('Loss')

plt.xlabel('N')

plt.legend()

plt.show()