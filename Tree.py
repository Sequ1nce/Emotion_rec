#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pandas as pd
import numpy as np
import matplotlib as plt


# In[2]:


import librosa
import librosa.display
import time
from tqdm import tqdm


# In[3]:


from sklearn.utils import shuffle
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder


# In[4]:


import tensorflow as tf
from matplotlib.pyplot import specgram
import keras
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Input, Flatten, Dropout, Activation
from keras.layers import Conv1D, MaxPooling1D, AveragePooling1D
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import confusion_matrix


# In[69]:


rnewdf=pd.read_csv('/home/bigfloppa/Рабочий стол/Datasets/Balanced.csv')
rnewdf=rnewdf.drop('Unnamed: 0',axis=1)


# In[70]:


Y=rnewdf['class']
rnewdf=rnewdf.drop('class',axis=1)


# In[71]:


rnewdf


# In[72]:


mean = np.mean(rnewdf, axis=0)
std = np.std(rnewdf, axis=0)

rnewdf = (rnewdf - mean)/std
rnewdf = (rnewdf - mean)/std

rnewdf


# In[87]:


for i in range(60,180):
    rnewdf=rnewdf.drop(str(i),axis=1)
rnewdf


# In[98]:


rnewdf=pd.concat([rnewdf,Y],axis=1)
rnewdf=rnewdf.sample(frac=1)
rnewdf=rnewdf.drop('class',axis=1)


# In[73]:


# for i in range(700):
#     if Y.iloc[i]=='-':
#         rnewdf.iloc[i].plot(color='r')
#     if Y.iloc[i]=='n':
#         rnewdf.iloc[i].plot(color='b')
#     if Y.iloc[i]=='+':
#         rnewdf.iloc[i].plot(color='g')


# In[99]:


import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import precision_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix


# In[100]:


X_train,X_test,y_train,y_test=train_test_split(rnewdf,Y,test_size=0.2)
y_train.shape


# In[101]:


rf=RandomForestClassifier(n_jobs=-1)
parameters={'n_estimators':range (20,30,2),'max_depth':range (5,10),'min_samples_leaf':range(1,7), \
           'min_samples_split':[2,4]}
gscv_rf=GridSearchCV(rf,param_grid=parameters,cv=3,n_jobs=-1)
gscv_rf.fit(X_train,np.ravel(y_train))
best_rf=gscv_rf.best_estimator_
best_rf


# In[102]:


result=best_rf.predict(X_test)
result


# In[103]:


np.array(y_test)


# In[104]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test,result)


# In[105]:


from sklearn.metrics import confusion_matrix


# In[106]:


confusion_matrix(y_test,result)


# In[107]:


pd.DataFrame(best_rf.feature_importances_).plot()


# In[ ]:




