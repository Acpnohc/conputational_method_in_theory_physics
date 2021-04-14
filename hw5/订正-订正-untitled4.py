# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 05:41:25 2021

@author: JChonpca_Huang
"""


# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 05:22:25 2021

@author: JChonpca_Huang
"""


# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 04:51:59 2021

@author: JChonpca_Huang
"""


# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 15:28:56 2021

@author: JChonpca_Huang
"""


import numpy as np
import matplotlib.pyplot as plt

# init

h = 0.01

step = 1000

L = 10

def kp_cal(t):
    
    
    tmp = r_matrix[:,t].copy().tolist()
        
        
    a = []
    
    for i in range(len(tmp)-1):
        
        if i == 0:
            
            tmp_grad = (1/2)*(tmp[i]+ L-tmp[-1]-1)**2 + (1/4)*(tmp[i]+ L-tmp[-1]-1)**4
        
        else:
            
            tmp_grad = (1/2)*(tmp[i+1]-tmp[i]-1)**2 + (1/4)*(tmp[i+1]-tmp[i]-1)**4
        
        a.append(tmp_grad)
    
    kp = np.array(a).sum()
    
    return kp




r = np.linspace(1,10,10).reshape(10) - 0.5

r_matrix = np.zeros([10,step+1])

r_matrix[:,0] = r

kp = kp_cal(0)

v = (np.random.uniform(size=10)).reshape(10)

v = v - v.mean()

v = v * np.sqrt((1-kp)/((1/2)*((v**2).sum())))

p_matrix = v_matrix = np.zeros([10,step+1])

p_matrix[:,0] = v_matrix[:,0] = v


def f(t):
    
    
    tmp = r_matrix[:,t].copy().tolist()
        
    a = []
    
    for i in range(len(tmp)):
        
        if i == 0:
            
            tmp_grad = (tmp[i+1]-tmp[i]-1)+(tmp[i+1]-tmp[i]-1)**3-(tmp[i]+L-tmp[-1]-1)-(tmp[i]+L-tmp[-1]-1)**3
        
        elif i == 9:
            
            tmp_grad = (tmp[0]+L-tmp[i]-1)+(tmp[0]+L-tmp[i]-1)**3-(tmp[i]-tmp[i-1]-1)-(tmp[i]-tmp[i-1]-1)**3
            
        else:
            
            tmp_grad = (tmp[i+1]-tmp[i]-1)+(tmp[i+1]-tmp[i]-1)**3-(tmp[i]-tmp[i-1]-1)-(tmp[i]-tmp[i-1]-1)**3
            
            
        a.append(tmp_grad)
    
    return [np.array(a).reshape(10)]

ek = []
ep = []

for i in range(step):
    
    if i == 0:        
        
        p_0 = p_matrix[:,0] - h*f(0)[0]
        
        r_0 = r_matrix[:,0] - h*p_matrix[:,0]
    
        p_matrix[:,1] = p_0 + 2*h*f(0)[0]
        
        r_matrix[:,1] = r_0 + 2*h*p_matrix[:,0]
                
    else:
        
        p_matrix[:,i+1] = p_matrix[:,i-1] + 2*h*f(i)[0]
        
        r_matrix[:,i+1] = r_matrix[:,i-1] + 2*h*p_matrix[:,i]
        

    ek.append(((1/2)*((v_matrix[:,i+1]**2).sum())))
    
    ep.append(kp_cal(i+1))


e = []

p = []

for i in range(step):
    
    e.append(kp_cal(i+1) + ((1/2)*((v_matrix[:,i+1]**2).sum())))
    
    p.append(v_matrix[:,i+1].sum())

time1 = np.linspace(1,step,step)*h

time2 = np.linspace(0,step,step+1)*h

plt.plot(time1,e,label='E')

plt.plot(time1,ek,label='Ek')

plt.plot(time1,ep,label='Ep')

plt.plot(time1,p,label='p')

plt.xlabel('Step')

plt.ylabel('Energy')

plt.legend()

plt.show()

for i in range(10):
    
    plt.plot(time2,r_matrix[i,:],label=str(i+1))

plt.xlabel('Time')

plt.ylabel('r')

plt.legend()

plt.show()