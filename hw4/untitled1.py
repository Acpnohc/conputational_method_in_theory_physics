#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 07:23:08 2021

@author: iluvatar
"""

import matplotlib.pyplot as plt
import numpy as np
import random

def fa(t):
    
    return np.cos(2*np.pi*t)

def fb(t):
    try:       
        return np.random.random(t.shape[0])    
    except:       
        return random.random()

def autocor(f,a):
    
    autx = [0]
    
    auty = [1]

    lins = np.linspace(0,a,1000)
    
    for i in lins:
        
        if i == 0:
            
            pass
        
        else:
            
            autx.append(i)
                        
            tmp_y2 = 0
            
            tx = np.linspace(0,10,1000)
            
            ty = f(tx)
            
            ty2 = ty**2
            
            mean_ty = ty.mean()
            
            mean_ty2 = ty2.mean()
            
            counter = 0
            
            for j in tx:
                
                tmp_y2 += f(j)*f(j+i)
                
                counter += 1
            
            mean_fty2 = tmp_y2/counter
                       
            auty.append((mean_fty2-mean_ty)/(mean_ty2-mean_ty))
            
    return autx,auty


x = np.linspace(0,10,1000)
y = fa(x)
plt.plot(x,y,label='raw data')
xx, yy = autocor(fa,10)
plt.plot(xx, yy,label='autocor')
plt.xlabel('t')
plt.ylabel('A or autocor')
plt.legend()
plt.show()


'''

def autocor(f,a):
    
    autx = [0]
    
    auty = [1]

    lins = np.linspace(0,a,1000)
    
    for i in lins:
        
        if i == 0:
            
            pass
        
        else:
            
            autx.append(i)
                        
            tmp_y2 = 0
            
            tx = np.linspace(0,1000,10000)
            
            ty = f(tx)
            
            ty2 = ty**2
            
            mean_ty = ty.mean()
            
            mean_ty2 = ty2.mean()
            
            counter = 0
            
            for j in tx:
                
                tmp_y2 += f(j)*f(j+i)
                
                counter += 1
            
            mean_fty2 = tmp_y2/counter
                       
            auty.append((mean_fty2-mean_ty)/(mean_ty2-mean_ty))
            
    return autx,auty



x = np.linspace(0,10,1000)
y = fb(x)
plt.plot(x,y,label='raw data')
xx, yy = autocor(fb,10)
plt.plot(xx, yy,label='autocor')
plt.xlabel('t')
plt.ylabel('B or autocor')
plt.legend()
plt.show()
'''
