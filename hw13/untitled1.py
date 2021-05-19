# -*- coding: utf-8 -*-
"""
Created on Wed May 19 13:40:25 2021

@author: iluvatar
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
import math

def f(x):
    
    if -math.pi/2 < x < math.pi/2:
        
        return (np.cos(x))**2
    
    else:
        
        return 0
    

def plt_pdf(p,low,high,dt,h):
    
    size = int((high-low)/h)
    
    x = np.linspace(0,1,size)
    
    x = low + x*(high-low)
    
    y = np.zeros(size)
    
    for i in range(size):
        
        y[i] = ( p.size - (p < x[i] - dt/2).sum() - (p > x[i] + dt/2).sum() )/p.size/dt
    
    # y[0] = y[0]*2
    
    # y[-1] = y[-1]*2
    
    return [x,y]





def loss(a,n):
    
    p = []

    p.append(-1 + 2*random.random())

    for i in range(1,n):
        
        
        # new = p[-1] -1 + 2*random.random()
        
        # new = -1 + 2*random.random()
        
        new = -a + 2*a*random.random()
        
        if f(new)/f(p[i-1]) > 1:
            
            p.append(new)
        
        else:
            
            if random.random() < f(new)/f(p[i-1]):
                
                p.append(new)
                
                
            else:
                
                p.append(p[-1])
        
    p = np.array(p)
    
    s = plt_pdf(p,min(p),max(p),0.1,0.01)

    # plt.plot(s[0],s[1],label='numerical')

    # xx = np.linspace(-math.pi/2,math.pi/2,10000)

    yy = 2*(np.cos(s[0]))**2/math.pi
    
    loss = (yy-s[1]).sum()
    
    return loss

xx = np.linspace(math.pi/2,10,100)

yy = []

yyy = []

yyyy = []

yyyyy = []

for i in xx:
    
    yy.append(loss(i,10))
    
for i in xx:
    
    yyy.append(loss(i,100))
    
for i in xx:
    
    yyyy.append(loss(i,1000))
    
for i in xx:
    
    yyyyy.append(loss(i,1000))

plt.plot(xx,yy,label='N=10')
plt.plot(xx,yyy,label='N=100')
plt.plot(xx,yyyy,label='N=1000')
plt.plot(xx,yyyy,label='N=1000')

plt.ylabel('Loss')

plt.xlabel('Width')

plt.legend()

plt.show()
