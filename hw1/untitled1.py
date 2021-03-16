# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 20:35:27 2021

@author: JChonpca_Huang
"""



import numpy as np
import matplotlib.pyplot as plt
import math
 
np.random.seed(0)
 
r = 1.0
a, b = (0., 0.)
 
x_min, x_max = a-r, a+r
y_min, y_max = b-r, b+r

xx = []

yy = []

zz = []

for n in range(100,100000,100):
    
    
 
    x = np.random.uniform(x_min, x_max, n) # 均匀分布
    y = np.random.uniform(y_min, y_max, n)
     
    d = np.sqrt((x-a)**2 + (y-b)**2)
     
    ind = np.where(d <= r, 1, 0)
    res = sum(ind)
     
    pi = 4 * res / n
    
    xx.append(n)
    
    yy.append(pi)
    
    zz.append(math.pi)



plt.scatter(xx,yy,s=5)

plt.plot(xx,zz,c='r')

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


plt.xlabel("随机点个数N") 

plt.ylabel("圆周率估计值") 

plt.show()