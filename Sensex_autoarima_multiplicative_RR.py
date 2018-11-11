
# coding: utf-8

# In[1]:


import pyramid; print("Pyramid Version---", pyramid.__version__)


# In[2]:


import pip; print("Pip Version ----", pip.__version__)


# In[3]:


import numpy as np
import scipy as sp


# In[4]:


import warnings
import pandas as pd
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib.pylab import rcParams
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
import pyramid as pm
from sklearn import metrics
from pyramid.arima import auto_arima
warnings.filterwarnings('ignore')


# In[15]:


from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity ="all"


# In[23]:


flight_data=pd.read_csv("bse.autoarima.csv")
flight_data.head()
flight_data.tail()
flight_data.shape
flight_data.describe()
flight_data.info()


# In[25]:


#Create a data range variable which captures data rane of the above data
month =pd.date_range('20171027', periods=248, freq='D')
month


# In[26]:


#Insert the datetime column in the original data
flight_data['datestamp']=month
flight_data.head()


# In[27]:


#select only datetime and # passengers data using loc
data=flight_data.loc[:,('datestamp','Close')]
data.head()
data.describe()


# In[28]:


#Reindex using method "set_index" the data on datetime variable
data.set_index('datestamp', inplace=True)
data.head()


# In[31]:


#plot the timeseries data, add x label, y label and title of the plot
plt.figure(figsize=(15,10))
plt.plot(data)
plt.xlabel ('Time')
plt.ylabel ('Close')
plt.title ('# Daily Close')
plt.show()


# In[32]:


#decompose timeseries to trend, seasonal and random components using multiplicative model
decomposition=seasonal_decompose(data,model='multiplicative')


# In[33]:


#plot trend of the series
plt.figure(figsize=(15,10))

trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

plt.subplot(221)
plt.plot(data, 'r', label='Original')
plt.legend(loc='best')
plt.subplot(222)
plt.plot(trend, 'b', label='Trend')
plt.legend(loc='best')
plt.subplot(223)
plt.plot(seasonal, 'g', label='Seasonality')
plt.legend(loc='best')
plt.subplot(224)
plt.plot(residual, 'y', label='Residuals')
plt.legend(loc='best')
plt.tight_layout()
plt.show()


# In[34]:


#test for the stationarity fo the time series
from pyramid.arima.stationarity import ADFTest
adf_test = ADFTest(alpha=0.05)
adf_test.is_stationary(data)


# In[60]:


#split the data in train and test datasets. Here in time series we cant give 70/30 percentage split.
train, test=data[:200], data[200:]
train.shape
test.shape


# In[61]:


plt.plot(train)
plt.plot(test)
plt.show()


# In[62]:


#fitting a stepwise model:
from pyramid.arima import auto_arima

Arima_model = auto_arima(train, start_p=1, start_q=1, max_p=3, max_q=3, start_P=0, start_Q=0, max_P=5, max_Q=5, m=12,
                         seasonal=True, trace=True, d=1, D=1, error_action='warn',suppress_warnings=True, stepwise=True, 
                         random_state=20, n_fits=30)
Arima_model.summary()


# In[64]:


prediction = pd.DataFrame(Arima_model.predict(n_periods=48), index =test.index)
prediction.columns = ['Predicted_Close']


# In[65]:


prediction


# In[66]:


test


# In[67]:


plt.figure(figsize=(15,10))
plt.plot(train, label='Training')
plt.plot(test, label = 'Test')
plt.plot(prediction, label='Predicted')
plt.legend(loc='upper center')
plt.show()


# In[68]:


test['Predicted_Close']=prediction
test['Error']= test['Close']-test['Predicted_Close']
test


# In[69]:


metrics.mean_absolute_error(test.Close, test.Predicted_Close)


# In[70]:


metrics.mean_squared_error(test.Close, test.Predicted_Close)


# In[72]:


metrics.median_absolute_error(test.Close, test.Predicted_Close)


# In[73]:


plt.figure(figsize=(20,10))
plt.subplot(121)
plt.plot(test.Error, color='#ff33CC')
plt.title('Error Distribution Over Time')
plt.subplot(122)
scipy.stats.probplot(test.Error, plot=plt)
plt.show


# In[74]:


plt.figure(figsize=(20,10))
pm.autocorr_plot(test.Error)
plt.show()

