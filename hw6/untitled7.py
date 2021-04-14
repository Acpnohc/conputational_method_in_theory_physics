# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 10:08:49 2021

@author: JChonpca_Huang
"""


# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 07:09:23 2021

@author: JChonpca_Huang
"""


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
import math

# init

h = 0.01

step = 100000

L = 10

def estimated_autocorrelation(x):
    
    n = len(x)
    
    variance = x.var()
    
    x = x-x.mean()
    
    r = np.correlate(x, x, mode = 'full')[-n:]
    
    assert np.allclose(r, np.array([(x[:n-k]*x[-(n-k):]).sum() for k in range(n)]))
    
    result = r/(variance*(np.arange(n, 0, -1)))
    
    return result


def plt_pdf(p,low,high,dt,h):    

    size = int((high-low)/h)    

    x = np.linspace(0,1,size)    

    x = low + x*(high-low)    

    y = np.zeros(size)    

    for i in range(size):        

        y[i] = ( p.size - (p < x[i] - dt/2).sum() - (p > x[i] + dt/2).sum() )/p.size/dt    

    return [x,y]

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

# M V P

m = np.array([1,math.sqrt(2)]*5)

r = np.linspace(1,L,10).reshape(10) - 0.5

r_matrix = np.zeros([10,step+1])

v_matrix = np.zeros([10,step+1])

p_matrix = np.zeros([10,step+1])

r_matrix[:,0] = r

kp = kp_cal(0)

v = (np.random.uniform(size=10)).reshape(10)

p = np.zeros(10).reshape(10)

for i in range(10):
    
    p[i] = m[i]*v[i]

v = v - p.mean()/m.mean()

v = v * np.sqrt((1-kp)/((1/2)*m.mean()*((v**2).sum())))

p = np.zeros(10).reshape(10)

for i in range(10):
    
    p[i] = m[i]*v[i]

v_matrix[:,0] = v

p_matrix[:,0] = p

def f(t): # 动量变化 = 合力
        
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


for i in range(step):
    
    for j in range(10):
                    
        r_matrix[j,i+1] = r_matrix[j,i] + h*v_matrix[j,i] +  (1/2)*(f(i)[0][j]/m[j])*(h**2)
    
    for j in range(10):
        
        v_matrix[j,i+1] = v_matrix[j,i] + (h/(2*m[j]))*(f(i)[0][j] + f(i+1)[0][j])
    
    for j in range(10):
        
        p_matrix[j,i+1] = v_matrix[j,i+1]*m[j]

e = []

p = []

ek = []

ep = []

for i in range(step):
    
    pp = 0
    
    ekk = 0
    
    for j in range(10):
        
        pp += m[j]*v_matrix[j,i+1]
        
        ekk += (1/2)*m[j]*(v_matrix[j,i+1]**2)
        
    
    kpp = kp_cal(i+1)
    
    e.append(kpp+ekk)
    
    p.append(pp)
    
    ek.append(ekk)
    
    ep.append(kpp)
    

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

code_1_v = v_matrix[0,:]

code_2_v = v_matrix[1,:]

s1 = plt_pdf(code_1_v,min(code_1_v),max(code_1_v),0.1,0.01)

plt.plot(s1[0],s1[1])

plt.show()

s2 = plt_pdf(code_2_v,min(code_2_v),max(code_2_v),0.1,0.01)

plt.plot(s2[0],s2[1])

plt.show()

m = estimated_autocorrelation(code_1_v)

plt.plot(np.linspace(0,500,50000),m[0:50000])

plt.xlabel('Lag')

plt.ylabel('Autocor')

plt.show()

n = estimated_autocorrelation(code_2_v)

plt.plot(np.linspace(0,500,50000),n[0:50000])

plt.xlabel('Lag')

plt.ylabel('Autocor')

plt.show()

mm = 0.5*math.sqrt(2)*(code_2_v**2)

nn = 0.5*1*(code_1_v**2)

mmm = estimated_autocorrelation(mm)

plt.plot(np.linspace(0,500,50000),mmm[0:50000])

plt.xlabel('Lag')

plt.ylabel('Autocor')

plt.show()

nnn = estimated_autocorrelation(nn)

plt.plot(np.linspace(0,500,50000),nnn[0:50000])

plt.xlabel('Lag')

plt.ylabel('Autocor')

plt.show()