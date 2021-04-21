# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 10:56:04 2021

@author: iluvatar
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 15:57:06 2021

@author: iluvatar
"""

import numpy as np
import matplotlib.pyplot as plt

# init

h = 0.01

step = 100000

L = 10

Par_Size = 10

T = 1

kb = 1

Nc = 10

def kp_cal(t):
    
    
    tmp = r_matrix[:,t].copy().tolist()
        
        
    a = []
    
    for i in range(len(tmp)):
        
        if i == Par_Size-1:
            
            tmp_grad = (1/2)*(tmp[0]+ L-tmp[-1]-1)**2 + (1/4)*(tmp[0]+ L-tmp[-1]-1)**4
        
        else:
            
            tmp_grad = (1/2)*(tmp[i+1]-tmp[i]-1)**2 + (1/4)*(tmp[i+1]-tmp[i]-1)**4
        
        a.append(tmp_grad)
    
    kp = np.array(a).sum()
    
    return kp




r = np.linspace(1,L,Par_Size).reshape(Par_Size) - 0.5

r_matrix = np.zeros([Par_Size,step+1])

r_matrix[:,0] = r

kp = kp_cal(0)

v = (np.random.uniform(size=Par_Size)).reshape(Par_Size)

v = v - v.mean()

v = v * np.sqrt((1-kp)/((1/2)*((v**2).sum())))

p_matrix = v_matrix = np.zeros([Par_Size,step+1])

p_matrix[:,0] = v_matrix[:,0] = v

N_sigma_matrix = np.zeros([Par_Size,step+2]) 


def f(t): # 动量变化 = 合力 = ma = a
    
    tmp = r_matrix[:,t].copy().tolist()
    
    tmp_ = v_matrix[:,t].copy().tolist()
    
    tmp__ = N_sigma_matrix[:,t+1].copy().tolist()
        
    a = []
    
    b = []
    
    for i in range(len(tmp)):
        
        
        
        if i == 0:
            
            tmp_grad = (tmp[i+1]-tmp[i]-1)+(tmp[i+1]-tmp[i]-1)**3-(tmp[i]+L-tmp[-1]-1)-(tmp[i]+L-tmp[-1]-1)**3 - tmp_[i]*tmp__[i]
        
        elif i == Par_Size-1:
            
            tmp_grad = (tmp[0]+L-tmp[i]-1)+(tmp[0]+L-tmp[i]-1)**3-(tmp[i]-tmp[i-1]-1)-(tmp[i]-tmp[i-1]-1)**3 - tmp_[i]*tmp__[i]
            
        else:
            
            tmp_grad = (tmp[i+1]-tmp[i]-1)+(tmp[i+1]-tmp[i]-1)**3-(tmp[i]-tmp[i-1]-1)-(tmp[i]-tmp[i-1]-1)**3 - tmp_[i]*tmp__[i]
            
            
        a.append(tmp_grad)
        
        b.append(Nc*((tmp_[i]**2)/(kb*T)-1))
    
    return [np.array(a).reshape(Par_Size),np.array(b).reshape(Par_Size)]

def p_cal(t):
    
    tmp = r_matrix[:,t].copy().tolist()
        
    a = []
    
    for i in range(len(tmp)):
        
        if i == Par_Size -1:
            
            tmp_grad = (tmp[0]+ L-tmp[-1]-1) + (tmp[0]+ L-tmp[-1]-1)**3
                    
        else:
            
            tmp_grad = (tmp[i+1]-tmp[i]-1)+(tmp[i+1]-tmp[i]-1)**3
            
            
        a.append(tmp_grad)
    
    return np.array(a).mean()


for i in range(step):
    
    N_sigma_matrix[:,i+1] = N_sigma_matrix[:,i] + h*f(i)[1]
    
    r_matrix[:,i+1] = r_matrix[:,i] + h*v_matrix[:,i] +  (1/2)*f(i)[0]*(h**2)
    
    v_matrix[:,i+1] = v_matrix[:,i] + h*(f(i)[0] + f(i+1)[0])/2
    
    
    
    # if (i+1)%1000 == 0:
        
    # v_matrix[:,i+1] = v_matrix[:,i+1]*np.sqrt((Par_Size*kb*T)/((v_matrix[:,i+1]**2).sum()))

        


ek = []

ep = []

e = []

p = []

pp = []

for i in range(step):
    
    e.append(kp_cal(i+1) + ((1/2)*((v_matrix[:,i+1]**2).sum())))
    
    p.append(v_matrix[:,i+1].sum())
    
    pp.append(p_cal(i+1))
    
    ek.append(((1/2)*((v_matrix[:,i+1]**2).sum())))
    
    ep.append(kp_cal(i+1))

time1 = np.linspace(1,step,step)*h

time2 = np.linspace(0,step,step+1)*h

plt.plot(time1,e,label='E')

plt.plot(time1,ek,label='Ek')

plt.plot(time1,ep,label='Ep')

plt.plot(time1,p,label='p')

plt.plot(time1,pp,label='pe')

plt.xlabel('Time')

plt.ylabel('Energy')

plt.legend()

plt.show()

for i in range(Par_Size):
    
    plt.plot(time2,r_matrix[i,:],label=str(i+1))

plt.xlabel('Time')

plt.ylabel('r')

plt.legend()

plt.show()