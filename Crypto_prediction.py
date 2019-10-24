import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
from dateutil.relativedelta import relativedelta
import seaborn as sns
import statsmodels.api as sm
from statsmodels.tsa.stattools import acf
from statsmodels.tsa.stattools import pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller

# Function to test the stationarity
# It is stationary if the "Test Statistic" is greater than the Critical Values (in absolute)
def test_stationarity(series):
    dftest = adfuller(series, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)

# Loading data from the csv file
df = pd.read_csv("Crypto.csv", index_col=0)
df.index.name = None
df.reset_index(inplace=True)

# Defining the datetimes of the time series.
start =  datetime.datetime.strptime(df['index'][0], "%Y-%m-%d %X")
date_list = [start + relativedelta(minutes=5*x) for x in range(0, 114)]

df['index'] = date_list
df.set_index(['index'], inplace=True)
df.index.name = None
df.columns= ['Crypto']

# Plotting data
df.Crypto.plot(figsize=(12,8), title= 'Crypto value', fontsize=14)
plt.savefig('Crypto.png', bbox_inches='tight')

# Testing stationarity using test_stationarity
test_stationarity(df.Crypto)


# Testing stationarity of first difference
df['first_difference'] =  df.Crypto - df.Crypto.shift(1)
test_stationarity(df.first_difference.dropna(inplace=False))

# Testing the existance of a season
test1 = [df.Crypto[12*i] for i in range(0,114//12)]
test_stationarity(test1)

# Testing the existance of a seasonal first difference
test2 = [test1[i] - test1[i+1]  for i in range(0,114//12-1)]
test_stationarity(test2)

# Running model with the parameters found from above
t = np.array([df.Crypto[x] for x in range(0,114)])
mod = sm.tsa.statespace.SARIMAX(t, trend='n', order=(0,1,0), seasonal_order=(1,1,1,12))

results = mod.fit()


# Prediction of 12 extra points (an hour)
predict_start = date_list[-1] + relativedelta(minutes=5)
predict_date_list = [predict_start + relativedelta(minutes=5*x) for x in range(0, 12)]
print(len(predict_date_list))
future = pd.DataFrame(index=predict_date_list, columns= df.columns)
df = pd.concat([df, future])

# Plotting predictions
predict = np.array(results.predict(start = 114, end = 125, dynamic= True))
np.array(df.Crypto[:114])

df['forecast'] = np.concatenate((np.array([0 for i in range(0,114)]) , predict))
df[['Crypto', 'forecast']].plot(figsize=(12, 8))
plt.savefig('Crypto_predict.png', bbox_inches='tight')

# Outputting forecast
for x in range(0, len(predict_date_list)):
    print(predict_date_list[x],df.forecast[x+114])
