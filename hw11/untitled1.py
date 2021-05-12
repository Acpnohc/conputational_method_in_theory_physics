# -*- coding: utf-8 -*-
"""
Created on Wed May  5 13:30:43 2021

@author: JChonpca_Huang
"""

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

def Eur(fv,h,t,x0,v0,T):
    
    global noise
    
    XX = np.zeros(int(t/h)+1)
    XX[0] = x0
    
    VV = np.zeros(int(t/h)+1)
    VV[0] = v0
    
    noise = np.random.normal(loc=0,scale=1,size=int(t/h))*np.sqrt(2*T*h)
    
    for i in range(1,int(t/h)+1):
        
        VV[i] = VV[i-1] + fv(XX[i-1], VV[i-1])*h + noise[i-1]
                
        XX[i] = XX[i-1] + VV[i-1]*h
        
    return [XX,VV]

#xx = []
#
#vv = []
#
#for i in range(10**5):
#    
#    p = Eur(f_sode,0.01,100,0,1,1)
#    
#    xx.append(p[0])
#    
#    vv.append(p[1])     