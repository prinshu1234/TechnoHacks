#!/usr/bin/env python
# coding: utf-8

# #  TechnoHacks Data Analytics Intership

# # Task 1 : Perform data cleaning

# In[55]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# Load the dataset (Gender submission)


# In[56]:


df = pd.read_csv(r"D:\intership\gender_submission.csv")


# In[57]:


df.shape


# In[58]:


df.head()


# In[59]:


df.dtypes


# In[60]:


df.isnull


# In[61]:


df.isnull().sum()


# In[9]:


df


# In[62]:


# load the dataset ( Training dataset)


# In[63]:


df = pd.read_csv(r"D:\intership\train.csv")


# In[64]:


df.shape


# In[65]:


df.head()


# In[66]:


df.dtypes


# In[67]:


df.isnull


# In[68]:


df.isnull().sum()


# In[17]:


# After sum of the null value we will see that there are two columns is null in our training data Age and Cabin we will remove it.


# In[69]:


df.isnull().sum()/891


# In[70]:


df.dropna(how = 'all') # Remove missing values from rows


# In[71]:


df.dropna(axis = 1) #Remove missing values from columns


# In[72]:


df


# In[73]:


df.drop(columns = ['Age','Cabin'])


# In[52]:


# outliers


# In[74]:


df['Age'].plot.box()


# In[24]:


# Load the dataset (Test dataset)


# In[76]:


df = pd.read_csv(r"D:\intership\test.csv")


# In[77]:


df.shape


# In[78]:


df.head()


# In[79]:


df.dtypes


# In[80]:


df.isnull


# In[81]:


df.isnull().sum()


# In[82]:


# After sum of the null value we will see that there are two columns is null in our test data Age and Cabin thenwe will remove it.


# In[83]:


df.isnull().sum()/418


# In[84]:


df.dropna(how = 'all') # Remove missing values from rows


# In[85]:


df.dropna(axis = 1) #Remove missing values from columns


# In[86]:


df


# In[87]:


df.drop(columns = ['Age','Cabin'])


# In[ ]:


#outlier


# In[88]:


df['Age'].plot.box()


# # Task 2 : Calculate summary statistics

# In[36]:


# Load the dataset


# In[37]:


df=pd.read_csv(r"C:\Users\prins\Downloads\sample-csv-file-for-testing.csv")


# In[38]:


df.shape


# In[39]:


df.head()


# In[40]:


df.columns


# In[41]:


df.dtypes


# In[42]:


df.describe() # Summry of the statistics


# # Task 3 : Visualization using histograms

# In[43]:


# Load the dataset


# In[44]:


df=pd.read_csv(r"C:\Users\prins\Downloads\Iris.csv")


# In[45]:


df.shape


# In[46]:


df.head()


# In[47]:


df.dtypes


# In[48]:


df['SepalLengthCm'].plot.hist()


# In[49]:


df['SepalLengthCm'].plot.hist()


# In[50]:


df['PetalLengthCm'].plot.hist()


# In[51]:


df['PetalWidthCm'].plot.hist()


# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings


# In[3]:


df = pd.read_csv("D:\intership\Iris.csv")


# In[4]:


df


# In[5]:


df.shape


# In[6]:


df.head()


# In[7]:


df.dtypes


# In[8]:


df.info()


# In[9]:


df.isnull().sum()


# In[10]:


df.describe()


# In[11]:


df.columns


# In[12]:


#Check duplicate values in the dataset
df[df.duplicated()].sum()


# In[13]:


df['Species'].value_counts()


# In[14]:


#The distribution of the Outcome variable
df['Species'].value_counts()*100/len(df)


# In[15]:


df['Species'].value_counts().plot.bar()    #categorical variable:- Bar Plot
plt.show()


# In[16]:


plt.figure(figsize=(10, 5))
sns.boxplot(data=df, orient="h")  # Create box plots for all columns
plt.show()


# In[61]:


# Continuous variable:- Histogram
plt.subplot(2,2,1)                   
plt.hist(df.SepalWidthCm, color = "Green")
plt.title('SepalWidthCm')


plt.subplot(2,2,2)
plt.hist(df.PetalWidthCm, color = "Yellow")
plt.title('PetalWidthCm')



plt.subplot(2,2,3)
plt.hist(df.SepalLengthCm, color = "Red")
plt.title('SepalLengthCm') 

plt.subplot(2,2,4)
plt.hist(df.PetalLengthCm, color = "black")
plt.title('PetalLengthCm')


plt.tight_layout()
plt.show()


# In[17]:


df = pd.read_csv("D:\intership\gender submission.csv")


# In[18]:


df


# In[21]:


(df.isnull().sum()/len(df)) * 100


# In[22]:


df.dropna(inplace=True)


# In[23]:


df


# In[24]:


plt.figure(figsize=(10, 5))
sns.boxplot(data=df, orient="h")  # Create box plots for all columns
plt.show()


# In[25]:


Q1 = df['Fare'].quantile(0.25)
Q3 = df['Fare'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

df= df[(df['Fare'] >= lower_bound) & (df['Fare'] <= upper_bound)]


# In[26]:


plt.figure(figsize=(10, 5))
sns.boxplot(data=df, orient="h")  # Create box plots for all columns
plt.show()


# In[80]:


plt.subplot(2,2,1)                        #Boxplot  
plt.boxplot(df.SepalWidthCm)
plt.title('SepalWidthCm')
plt.show()

plt.subplot(2,2,2)
plt.boxplot(df.PetalWidthCm)
plt.title('PetalWidthCm')
plt.show()


# In[79]:


plt.subplot(2,2,1)                        #Boxplot  
plt.boxplot(df.SepalWidthCm)
plt.title('SepalWidthCm')
plt.show()

plt.subplot(2,2,2)
plt.boxplot(df.PetalWidthCm)
plt.title('PetalWidthCm')
plt.show()


# In[77]:


plt.subplot(2,2,1)                        #Boxplot  
plt.boxplot(df.SepalWidthCm)
plt.title('SepalWidthCm')
plt.show()

plt.subplot(2,2,2)
plt.boxplot(df.PetalWidthCm)
plt.title('PetalWidthCm')
plt.show()


# In[69]:


plt.subplot(2,2,3)                                  #Boxplot
plt.boxplot(df.PetalLengthCm)
plt.title('PetalLengthCm')
plt.show()

plt.subplot(2,2,4)
plt.boxplot(df.SepalLengthCm)
plt.title('SepalLengthCm')
plt.show()


# In[70]:


#Z_SCORES                                    #Removing Outliers using z_score
outliers=[]
def detect_outliers(data):
    threshold=3
    mean=np.mean(data)
    std=np.std(data)
    for x in data:
        z_score=(x-mean)/std
        if np.abs(z_score)>threshold:
            outliers.append(x)
    return outliers


# In[71]:


detect_outliers(df.SepalWidthCm)


# In[73]:


df["SepalWidthCm"].plot.box()


# In[74]:


Q1 = df['SepalWidthCm'].quantile(0.25)
Q3 = df['SepalWidthCm'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

df= df[(df['SepalWidthCm'] >= lower_bound) & (df['SepalWidthCm'] <= upper_bound)]


# In[75]:


plt.subplot(2,2,1)                        #Boxplot  
plt.boxplot(df.SepalWidthCm)
plt.title('SepalWidthCm')
plt.show()

plt.subplot(2,2,2)
plt.boxplot(df.PetalWidthCm)
plt.title('PetalWidthCm')
plt.show()


# In[67]:


plt.hist(df.SepalLengthCm, color = "Darkblue")
plt.title('SepalLengthCm')
plt.show()


# In[66]:


df['SepalLengthCm'].plot.hist()
plt.title("SepalLengthCm")
plt.show()


# In[25]:


df['SepalWidthCm'].plot.box()
plt.title("SepalWidthCm")
plt.show()


# In[48]:


from scipy.stats import zscore

z_scores = zscore(df['SepalWidthCm'])
outliers = np.where(np.abs(z_scores) > 3)
print("Outliers in the 'SepalwidthCm' column:", df['SepalWidthCm'].iloc[outliers])


# In[49]:


df['SepalWidthCm'].plot.box()
plt.title("SepalWidthCm")
plt.show()


# In[ ]:





# In[47]:


sns.boxplot('Species','SepalLengthCm',data=df)


# In[50]:


plt.subplot(1,2,1)
plt.boxplot(df.SepalWidthCm)
plt.title('SepalWidthCm')

plt.subplot(1,2,2)
plt.boxplot(df.PetalWidthCm)
plt.title('PetalWidthCm')
plt.show()


# In[35]:


plt.subplot(1,2,1)
plt.boxplot(df.SepalWidthCm)
plt.title('SepalWidthCm')

plt.subplot(1,2,2)
plt.boxplot(df.PetalWidthCm)
plt.title('PetalWidthCm')
plt.show()


# In[32]:


plt.subplot(1,2,1)
plt.boxplot(df.SepalLengthCm)
plt.title('SepalLengthCm')

plt.subplot(1,2,2)
plt.boxplot(df.PetalLengthCm)
plt.title('PetalLengthCm')
plt.show()


# In[46]:


plt.subplot(2,2,1)
plt.hist(df.SepalWidthCm)
plt.title('SepalWidthCm')

plt.subplot(2,2,2)
plt.hist(df.PetalWidthCm)
plt.title('PetalWidthCm')


plt.subplot(2,2,3)
plt.hist(df.SepalLengthCm)
plt.title('SepalLengthCm')

plt.subplot(2,2,4)
plt.hist(df.PetalLengthCm)
plt.title('PetalLengthCm')
plt.tight_layout()
plt.show()


# In[23]:


# Use boxplots to visualize the outliers
plt.figure(figsize=(10, 4))
sns.boxplot(data=df, orient="v")  # Create box plots for all columns
plt.show()


# In[17]:


# Use Z-score to remove the outliers
from scipy.stats import zscore

z_scores = zscore(df['SepalWidthCm'])
outliers = np.where(np.abs(z_scores) > 3)
print("Outliers in the 'SepalwidthCm' column:", df['SepalWidthCm'].iloc[outliers])


# In[ ]:




