# -*- coding: utf-8 -*-
"""
Created on Wed May  5 13:11:05 2021

@author: JChonpca_Huang
"""


# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 19:03:14 2021
@author: JChonpca_Huang
"""

import numpy as np
import matplotlib.pyplot as plt


def f_sode(x,v):
    
    return  -v

def SRK2(fv,h,t,x0,v0,T):
    
    XX = np.zeros(int(t/h)+1)
    XX[0] = x0
    
    VV = np.zeros(int(t/h)+1)
    VV[0] = v0
    
    noise = np.random.normal(loc=0,scale=1,size=int(t/h))*np.sqrt(2*T*h)
    
    for i in range(1,int(t/h)+1):
        
        xk1 = VV[i-1]
        
        vk1 = fv(XX[i-1], VV[i-1])
        
        xk2 = VV[i-1] + h*vk1 + noise[i-1]
        
        vk2 = fv(XX[i-1] + xk1*h, VV[i-1] + h*vk1 + noise[i-1])
        
        VV[i] = VV[i-1] + (1/2)*h*(vk1+vk2) + noise[i-1]
                
        XX[i] = XX[i-1] + (1/2)*h*(xk1+xk2)
        
    return [XX,VV]

xx = []

vv = []

for i in range(10**4):
    
    print(i)
    
    p = SRK2(f_sode,0.01,100,0,1,1)
    
    xx.append(p[0])
    
    vv.append(p[1])
    
    
#
#plt.plot(p[0])
#
#plt.show()        