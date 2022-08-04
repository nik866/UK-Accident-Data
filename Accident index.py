#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
import pandas as pd


# In[2]:


import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


import string


# In[4]:


df = pd.read_csv('D:\\Excel data\\uk_accident_dataset.csv')


# In[5]:


df.head()


# In[6]:


df.describe()


# In[7]:


plt.scatter(df['Accident_ID'],df['Number_of_Casualties'])


# # Accident_ID vs Police_Force

# In[8]:


plt.scatter(df['Accident_ID'],df['Police_Force'])


# In[9]:


x = df[['Local_Authority_(District)','Number_of_Casualties']]
y = df['Police_Force']


# In[10]:


x


# In[11]:


y


# In[12]:


from sklearn.model_selection import train_test_split


# In[13]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)


# In[14]:


x_train


# In[15]:


len(x_test)


# In[16]:


y_test


# In[17]:


df.shape


# In[35]:


plt.scatter(df['country'],df['Number_of_Casualties'])


# In[38]:


import matplotlib.pyplot as plt
import numpy as np
from matplotlib import style


# In[ ]:


plt.pie(df['Police_Force'],df['Number_of_Casualties'])


# In[ ]:





# In[ ]:




