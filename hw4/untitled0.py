#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 17:17:16 2021

@author: iluvatar
"""

import matplotlib.pyplot as plt
import numpy as np
import random

def plt_pdf(p,low,high,dt,h):    
    size = int((high-low)/h)    
    x = np.linspace(0,1,size)    
    x = low + x*(high-low)    
    y = np.zeros(size)    
    for i in range(size):        
        y[i] = ( p.size - (p < x[i] - dt/2).sum() - (p > x[i] + dt/2).sum() )/p.size/dt    
    return [x,y]

path = []
for ip in range(1000,10000,1000):
    for i in range(100000):        
        tmp = [0]
        for j in range(ip):
            if random.random()>0.5:
                tmp.append(tmp[-1] + 1)
            else:
                tmp.append(tmp[-1] - 1)
        path.append(tmp)

# for k in path:
#     plt.plot(k)
# plt.xlabel('t')
# plt.ylabel('L')
# plt.legend()
# plt.show()

x = []
y = []
for m in range(9):

    pp = np.array(path[m*100000:(m+1)*100000])

    x.append(pp.shape[1])
    y.append((pp[:,-1]).mean())

plt.plot(x,y,label='simulation')
plt.plot([1000,9000],[0,0],label='theory')
plt.xlabel('t')
plt.ylabel('Mean')
plt.legend()
plt.show()

x = []
y = []
for m in range(9):
    
    pp = np.array(path[m*1000:(m+1)*1000])
    
    x.append(pp.shape[1])
    y.append((pp[:,-1]).var())

plt.plot(x,y,label='simulation')
plt.plot([1000,9000],[1000,9000],label='theory')
plt.xlabel('t')
plt.ylabel('Var')
plt.legend()
plt.show()

plt.subplots_adjust(left=None, bottom=0.1, right=None, top=None, wspace=0.5, hspace=0.5)
plt.rcParams['figure.figsize'] = (10, 5)

for m in range(9):
    
    plt.subplot(3,3,m+1) 
    pp = np.array(path[m*1000:(m+1)*1000])[:,-1]
    s = plt_pdf(pp,min(pp),max(pp),10,1)
    plt.plot(s[0],s[1],label='S-'+str(1000+1000*m))
    plt.plot(s[0],(1/np.sqrt(2*np.pi*(1000+1000*m)))*np.e**(-s[0]**2/(2*(1000+1000*m))),label='T-'+str(1000+1000*m))
    plt.title('T='+str(1000+1000*m))
    # plt.legend()
plt.show()