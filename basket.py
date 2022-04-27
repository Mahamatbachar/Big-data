#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv('baskets.csv')
df.head()


# In[2]:


df['basket_id'].value_counts()


# In[3]:


df.groupby("basket_id").product.unique()


# In[4]:


df['product'].value_counts()


# In[5]:


df.value_counts()[:50].plot(kind='bar', figsize=(15,5))


# In[6]:


df.groupby("basket_id").product.unique()


# In[7]:


# Print transactions header.
df.head()


# In[12]:


# check for missing values
df.isnull().sum()


# In[20]:


df.value_counts(normalize = True)[:10]


# In[27]:


df['product']= df['product'].str.strip() #removes spaces from beginning and end of sentences in the column 'product'
df.dropna(axis=0, subset=['basket_id'],inplace=True) #removes any duplicate 'Order' No.
df['basket_id']=df['basket_id'].astype('str')  #converting 'Order' No. to be string 
df = df[~df['basket_id'].str.contains('C')] #removing any credit Order No. if present any.
df.head()


# In[44]:


mybasket= (df.groupby(['basket_id','product']).sum())


# In[47]:


#viewing transaction basket
mybasket.head()


# In[48]:


def my_encode_units(x):
    if x <= 0:
        return 0
    if x>= 1:
        return 1
    
my_basket_sets =  mybasket.applymap(my_encode_units)


# In[49]:


from apyori import apriori
#generating frequent itemsets
my_frequent_itemsets = apriori(my_basket_sets, min_support=0.07, use_colnames=True)
#considering the rules that have 0.07 support
#my_frequent_itemsets => type of transactions


# In[50]:


#viewing top 100 rules
my_basket_sets.head(100)


# In[53]:


import pandas as pd

df1 = pd.read_csv('baskets.csv')
df2 = pd.read_csv('airroutes.csv')

data = pd.concat([df1,df2])
data.head()


# In[ ]:




