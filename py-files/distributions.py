#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt


# In[4]:


np.random.seed(0)
init_x = np.random.normal(0, np.sqrt(2), 50)


# In[45]:


plt.plot(init_x, np.zeros_like(init_x), 'kx')
plt.xlabel('Initialized x values')
plt.show()


# In[10]:


x_1 = [0.5*x + 25*x/(1+x**2) + 8*np.cos(0) + np.random.normal(0, np.sqrt(10)) for x in init_x]


# In[13]:


y_1 = [0.05*x**2 + np.random.normal(0, np.sqrt(1)) for x in x_1]


# In[47]:


plt.subplots(1,3,figsize=(15,3))

plt.subplot(131)
plt.plot(init_x, np.zeros_like(init_x), 'kx')
plt.xlabel('Initialized x values')

plt.subplot(132)
plt.plot(x_1, np.zeros_like(x_1), 'kx')
plt.xlabel('$\mathcal{A}(x_0) + \mathcal{N}(0,Q)$')

plt.subplot(133)
plt.plot(y_1, np.zeros_like(y_1), 'kx')
plt.xlabel('$\mathcal{C}(x_1) + \mathcal{N}(0,R)$')
plt.show()


# In[48]:


x_2 = [0.5*x + np.random.poisson(np.sqrt(10)) for x in init_x]
y_2 = [x + np.random.poisson(np.sqrt(1)) for x in x_2]


# In[49]:


plt.subplots(1,3,figsize=(15,3))

plt.subplot(131)
plt.plot(init_x, np.zeros_like(init_x), 'kx')
plt.xlabel('Initialized x values')

plt.subplot(132)
plt.plot(x_2, np.zeros_like(x_2), 'kx')
plt.xlabel('$\mathcal{A}(x_0) + Poisson(\lambda=Q)$')

plt.subplot(133)
plt.plot(y_2, np.zeros_like(y_2), 'kx')
plt.xlabel('$\mathcal{C}(x_1) + Poisson(\lambda=R)$')
plt.show()


# In[50]:


x_3 = [0.5*x + np.random.normal(0,np.sqrt(10)) for x in init_x]
y_3 = [x + np.random.normal(0,np.sqrt(1)) for x in x_3]


# In[51]:


plt.subplots(1,3,figsize=(15,3))

plt.subplot(131)
plt.plot(init_x, np.zeros_like(init_x), 'kx')
plt.xlabel('Initialized x values')

plt.subplot(132)
plt.plot(x_3, np.zeros_like(x_3), 'kx')
plt.xlabel('$\mathcal{A}(x_0) + Poisson(\lambda=Q)$')

plt.subplot(133)
plt.plot(y_3, np.zeros_like(y_3), 'kx')
plt.xlabel('$\mathcal{C}(x_1) + Poisson(\lambda=R)$')
plt.show()


# In[ ]:




