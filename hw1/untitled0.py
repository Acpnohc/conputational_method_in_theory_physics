# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 20:08:23 2021

@author: JChonpca_Huang
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
 
n = 5000
 
r = 1.0
a, b = (0., 0.)
 
x_min, x_max = a-r, a+r
y_min, y_max = b-r, b+r
 
x = np.random.uniform(x_min, x_max, n) # 均匀分布
y = np.random.uniform(y_min, y_max, n)
 
d = np.sqrt((x-a)**2 + (y-b)**2)
 
ind = np.where(d <= r, 1, 0)
res = sum(ind)
 
pi = 4 * res / n
 
print('pi: ', pi)
 

fig = plt.figure(figsize=(10,10))
 
axes = fig.add_subplot(111)
 
#axes.set_title('PI is: %f'  %pi)

for i in range(len(ind)):
	if ind[i] == 1:
		axes.plot(x[i], y[i],'ro',markersize = 2,color='r')
	else:
		axes.plot(x[i], y[i],'ro',markersize = 2,color='g')

plt.axis('equal')
 
circle = Circle(xy=(a,b), radius=r, alpha=0.5)
axes.add_patch(circle)
 
plt.show()
