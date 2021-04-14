# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 02:23:56 2021

@author: JChonpca_Huang
"""



import numpy as np
import matplotlib.pyplot as plt
import random

def fa(t):
    
    return np.cos(2*np.pi*t)

def fb(t):
    try:       
        return np.random.random(t.shape[0])    
    except:       
        return random.random()


x1 = np.linspace(0,10,10000)
y1 = fa(x1)

x2 = np.linspace(0,10,10000)
y2 = fb(x2)


def estimated_autocorrelation(x):
    
    n = len(x)
    
    variance = x.var()
    
    x = x-x.mean()
    
    r = np.correlate(x, x, mode = 'full')[-n:]
    
    assert np.allclose(r, np.array([(x[:n-k]*x[-(n-k):]).sum() for k in range(n)]))
    
    result = r/(variance*(np.arange(n, 0, -1)))
    
    return result

yy1 =  estimated_autocorrelation(y1)

yy2 =  estimated_autocorrelation(y2)

plt.plot(np.linspace(0,5,5000),yy1[0:5000])

plt.xlabel('Lag')

plt.ylabel('Autocor')

plt.show()

plt.plot(np.linspace(0,5,5000),yy2[0:5000])

plt.xlabel('Lag')

plt.ylabel('Autocor')

plt.show()