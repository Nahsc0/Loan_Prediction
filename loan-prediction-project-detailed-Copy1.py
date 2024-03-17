#!/usr/bin/env python
# coding: utf-8

# # Loan Prediction
# ## Binary Classification using Logistic Regression

# ![poster.jpg](attachment:poster.jpg)

# ### Importing Libraries

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import warnings
warnings.filterwarnings('ignore') 


# ### Importing & Loading the dataset

# In[2]:


df = pd.read_csv('train.csv')
df.head()


# ### Dataset Info:

# In[3]:


df.info()


# ### Dataset Shape:

# In[4]:


df.shape


# ## Data Cleaning

# ### Checking the Missing Values

# In[5]:


df.isnull().sum()


# #### First we will fill the Missing Values in "LoanAmount" & "Credit_History" by the 'Mean' & 'Median' of the respective variables.

# In[6]:


df['LoanAmount'] = df['LoanAmount'].fillna(df['LoanAmount'].mean())


# In[7]:


df['Credit_History'] = df['Credit_History'].fillna(df['Credit_History'].median())


# ### Let's confirm if there are any missing values in 'LoanAmount' & 'Credit_History'

# In[8]:


df.isnull().sum()


# ### Now, Let's drop all the missing values remaining.

# In[9]:


df.dropna(inplace=True)


# ### Let's check the Missing values for the final time!

# In[10]:


df.isnull().sum()


# Here, we have dropped all the missing values to avoid disturbances in the model. The Loan Prediction requires all the details to work efficiently and thus the missing values are dropped.

# ### Now, Let's check the final Dataset Shape

# In[11]:


df.shape


# ### Exploratory Data Analyis

# #### Comparison between Parameters in getting the Loan:

# In[ ]:





# In[12]:


plt.figure(figsize = (100, 50))
sns.set(font_scale = 5)
plt.subplot(331)
sns.countplot(x=df['Gender'],hue=df['Loan_Status'])

plt.subplot(332)
sns.countplot(x=df['Married'],hue=df['Loan_Status'])

plt.subplot(333)
sns.countplot(x=df['Education'],hue=df['Loan_Status'])

plt.subplot(334)
sns.countplot(x=df['Self_Employed'],hue=df['Loan_Status'])

plt.subplot(335)
sns.countplot(x=df['Property_Area'],hue=df['Loan_Status'])



# In[ ]:





# ### Let's replace the Variable values to Numerical form & display the Value Counts
# 
# The data in Numerical form avoids disturbances in building the model. 

# In[13]:


df['Loan_Status'].replace('Y',1,inplace=True)
df['Loan_Status'].replace('N',0,inplace=True)


# In[14]:


df['Loan_Status'].value_counts()


# In[15]:


df.Gender=df.Gender.map({'Male':1,'Female':0})
df['Gender'].value_counts()


# In[16]:


df.Married=df.Married.map({'Yes':1,'No':0})
df['Married'].value_counts()


# In[17]:


df.Dependents=df.Dependents.map({'0':0,'1':1,'2':2,'3+':3})
df['Dependents'].value_counts()


# In[18]:


df.Education=df.Education.map({'Graduate':1,'Not Graduate':0})
df['Education'].value_counts()


# In[19]:


df.Self_Employed=df.Self_Employed.map({'Yes':1,'No':0})
df['Self_Employed'].value_counts()


# In[20]:


df.Property_Area=df.Property_Area.map({'Urban':2,'Rural':0,'Semiurban':1})
df['Property_Area'].value_counts()


# In[21]:


df['LoanAmount'].value_counts()


# In[22]:


df['Loan_Amount_Term'].value_counts()


# In[23]:


df['Credit_History'].value_counts()


# In[ ]:





# From the above figure, we can see that **Credit_History** (Independent Variable) has the maximum correlation with **Loan_Status** (Dependent Variable). Which denotes that the Loan_Status is heavily dependent on the Credit_History.

# ### Final DataFrame

# In[24]:


df.head()


# ### Importing Packages for Classification algorithms

# In[25]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics


# ### Splitting the data into Train and Test set

# In[26]:


X = df.iloc[1:542,1:12].values
y = df.iloc[1:542,12].values


# In[27]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=0)


# ### Logistic Regression (LR)

# Logistic regression is a supervised learning classification algorithm used to predict the probability of a target variable. 
# 
# Mathematically, a logistic regression model predicts P(Y=1) as a function of X. It is one of the simplest ML algorithms that can be used for various classification problems such as spam detection, Diabetes prediction, cancer detection etc.

# #### Sigmoid Function

# ![logistic%20regression.png](attachment:logistic%20regression.png)

# In[28]:


model = LogisticRegression()
model.fit(X_train,y_train)

lr_prediction = model.predict(X_test)
print('Logistic Regression accuracy = ', metrics.accuracy_score(lr_prediction,y_test))


# In[29]:


print("y_predicted",lr_prediction)
print("y_test",y_test)


# **CONCLUSION:**
# 
# 1. The Loan Status is heavily dependent on the Credit History for Predictions.
# 2. The Logistic Regression algorithm gives us the maximum Accuracy (79% approx) compared to the other 3 Machine Learning Classification Algorithms.

# In[30]:


get_ipython().system('pip install joblib')


# In[31]:


import joblib


# In[32]:


# Assuming your trained model object is named 'model'
joblib.dump(model, 'model.pkl')


# In[33]:


# Assuming df is your DataFrame containing the dataset
feature_names = df.columns[1:12]
print("Features used to train the model:")
print(feature_names)


# In[ ]:




