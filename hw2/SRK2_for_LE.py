# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 19:03:14 2021

@author: JChonpca_Huang
"""

import numpy as np
import matplotlib.pyplot as plt


def f_sode(x,v):
    
    return  -x -v

def SRK2(fv,h,t,x0,v0,T):
    
    XX = np.zeros(int(t/h)+1)
    XX[0] = x0
    
    VV = np.zeros(int(t/h)+1)
    VV[0] = v0
    
    noise = np.random.normal(loc=0,scale=1,size=int(t/h))*np.sqrt(2*T*h)
    
    for i in range(1,int(t/h)+1):
        
        vk1 = fv(XX[i-1], VV[i-1])
        vk2 = fv(XX[i-1], VV[i-1] + h*vk1 + noise[i-1])
        
        VV[i] = VV[i-1] + (1/2)*h*(vk1+vk2) + noise[i-1]
                
        XX[i] = XX[i-1] + VV[i-1]*h
        
    return [XX,VV]

def E_stat(P):
    
    x = P[0][2000::]
    
    v = P[1][2000::]
    
    E = 0
    
    for i in range(len(x)):
        
        E = E + (1/2)*(x[i]**2 + v[i]**2)
    
    return E/len(x)

X = []
Y = []
P = []

for T in range(0,180):
    
    
     p = SRK2(f_sode,0.01,100,0,0,T)
     P.append(p)
     X.append(T)
     Y.append(E_stat(p))


plt.plot(X,Y)

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

plt.xlabel("T") 
plt.ylabel("Energy") 

plt.show()        