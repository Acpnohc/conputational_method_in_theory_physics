# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 15:28:56 2021

@author: JChonpca_Huang
"""


import numpy as np
import matplotlib.pyplot as plt
import copy

# init

h = 0.01

step = 1000

def kp_cal(t):
    

    tmp = r_matrix[:,t].copy()
    
    tmp_tmp_tmp = np.hstack([tmp-10,tmp,tmp+10])
    
    tmp = tmp.tolist()
    
    tmp_tmp_tmp = tmp_tmp_tmp.tolist()
        
        
    a = []
    
    for i in tmp:
        
        tmp_sort = copy.copy(tmp)
        
        tmp_sort = copy.copy(tmp_tmp_tmp)
        
        tmp_sort.sort()
        
        site = tmp_sort.index(i)
        
        tmp_grad = 0
        
        if site == len(tmp_sort)-1:
            
            tmp_grad = 0
        
        else:
            
            tmp_grad = (1/2)*(tmp_sort[site+1]-tmp_sort[site]-1)**2 + (1/4)*(tmp_sort[site+1]-tmp_sort[site]-1)**4
            
        a.append(tmp_grad)
    
    kp = np.array(a).sum()
    
    return kp


kkpp = []

for i in range(1000):
    
    r = (10*np.random.uniform(size=10)).reshape(10)
    
    #r = np.linspace(1,10,10).reshape(10)
    
#    r = np.linspace(1,10,10).reshape(10) + np.random.normal(0,0.1,10)
    
    r_matrix = np.zeros([10,step+1])
    
    r_matrix[:,0] = r
    
    kkpp.append(kp_cal(0))

plt.hist(kkpp)
plt.xlabel('Ep')
plt.ylabel('Counts')
plt.show()


x = []
kkpp = []

for i in range(501):
    
#    r = (10*np.random.uniform(size=10)).reshape(10)
    
    #r = np.linspace(1,10,10).reshape(10)
    
    r = np.linspace(1,10,10).reshape(10) + np.random.normal(0,i/1000,10)
    
    r_matrix = np.zeros([10,step+1])
    
    r_matrix[:,0] = r
    
    x.append(i/1000)
    
    kkpp.append(kp_cal(0))

plt.plot(x,kkpp)
plt.xlabel('Var')
plt.ylabel('Ep')
plt.show()

#r = (10*np.random.uniform(size=10)).reshape(10)

#r = np.linspace(1,10,10).reshape(10)

r = np.linspace(1,10,10).reshape(10) + np.random.normal(0,0.1,10)

r_matrix = np.zeros([10,step+1])

r_matrix[:,0] = r


kp = kp_cal(0)

v = (np.random.uniform(size=10)).reshape(10)

v = v - v.sum()/10

v = v * np.sqrt((1-kp)/((1/2)*((v**2).sum())))

v_matrix = np.zeros([10,step+1])

v_matrix[:,0] = v

def f(t):
        
    tmp = r_matrix[:,t].copy()
    
    tmp_tmp_tmp = np.hstack([tmp-10,tmp,tmp+10])
    
    tmp = tmp.tolist()
    
    tmp_tmp_tmp = tmp_tmp_tmp.tolist()
    
    a = []
    
    for i in tmp:
        
        tmp_sort = copy.copy(tmp_tmp_tmp)
        
        tmp_sort.sort()
        
        site = tmp_sort.index(i)
        
        tmp_grad = -4
        
        tmp_grad_1 = 0
        
        if tmp_sort[site-1] < 0:
            
            tmp_grad_1 = 2*min([abs(i-tmp_sort[site-1]),abs(i-tmp_sort[site-1]+10),abs(i-tmp_sort[site-1]+20)])
        
        elif 0<= tmp_sort[site-1] <=10:
            
            tmp_grad_1 = 2*min([abs(i-tmp_sort[site-1]),abs(i-tmp_sort[site-1]-10),abs(i-tmp_sort[site-1]+10)])
            
        elif tmp_sort[site-1] > 10:
            
            tmp_grad_1 = 2*min([abs(i-tmp_sort[site-1]),abs(i-tmp_sort[site-1]-10),abs(i-tmp_sort[site-1]-20)])
            
        tmp_grad_2 = 0
        
        if tmp_sort[site+1] < 0:
            
            tmp_grad_2 = 2*min([abs(tmp_sort[site+1]-i),abs(tmp_sort[site+1]+10-i),abs(tmp_sort[site+1]+20-i)])
        
        elif 0<= tmp_sort[site-1] <=10:
            
            tmp_grad_2 = 2*min([abs(i-tmp_sort[site+1]),abs(tmp_sort[site+1]-10-i),abs(tmp_sort[site+1]+10-i)])
            
        elif tmp_sort[site-1] > 10:
            
            tmp_grad_2 = 2*min([abs(i-tmp_sort[site+1]),abs(tmp_sort[site+1]-10-i),abs(tmp_sort[site+1]-20-i)])        
        
        tmp_grad = tmp_grad + tmp_grad_1 + tmp_grad_2
        
        a.append(tmp_grad)
    
    f_r = []
    
    f_v = []
    
    for i in a:
        
        site = a.index(i)
        
        f_r.append((1/2)*i*(h**2)+v_matrix[site,t+1])
        
        f_v.append(i*h)
        
    return [np.array(f_r).reshape(10),np.array(f_v).reshape(10)]

ek = []
ep = []

for i in range(step):    
    
    if i == 0:        
        
        r_0 = r_matrix[:,0] + h*f(0)[0]
        
        v_0 = v_matrix[:,0] + h*f(0)[1]
    
        r_matrix[:,1] = r_0 + 2*h*f(0)[0]
        
        v_matrix[:,1] = v_0 + 2*h*f(0)[1]
        
        v_matrix[:,1] = v_matrix[:,1] * np.sqrt((1-kp_cal(1))/((1/2)*((v_matrix[:,1]**2).sum())))
        
    else:
        
        r_matrix[:,i+1] = r_matrix[:,i-1] + 2*h*f(i)[0]
        
        v_matrix[:,i+1] = v_matrix[:,i-1] + 2*h*f(i)[1]
        
        v_matrix[:,i+1] = v_matrix[:,1] * np.sqrt((1-kp_cal(i+1))/((1/2)*((v_matrix[:,i+1]**2).sum())))
        
    
    print(kp_cal(i+1) + ((1/2)*((v_matrix[:,i+1]**2).sum())))
    
    ek.append(((1/2)*((v_matrix[:,i+1]**2).sum())))
    
    ep.append(kp_cal(i+1))

e = []

for i in range(step):
    
    e.append(kp_cal(i+1) + ((1/2)*((v_matrix[:,i+1]**2).sum())))


plt.plot(e,label='E')
plt.plot(ek,label='Ek')
plt.plot(ep,label='Ep')
plt.xlabel('Step')
plt.ylabel('Energy')
plt.legend()
plt.show()

for i in range(10):
    plt.plot(r_matrix[i,:],label=str(i+1))

plt.xlabel('step')
plt.ylabel('r')
plt.legend()
plt.show()