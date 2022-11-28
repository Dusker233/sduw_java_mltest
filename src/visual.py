#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv('./Real_Ans.csv')
df1 = pd.read_csv('./Predicted_Ans.csv')


# 下面是真实数据与预测数据的可视化图，其中红色点为真实数据，黑色点为预测数据
# 
# 如果预测正确，则黑色点与红色点应当重合，否则不会重合

# In[3]:


plt.scatter(df['Real_index'], df['Real_class'], 15, 'red')
plt.scatter(df1['Predicted_index'], df1['Predicted_class'], 15, 'black')

