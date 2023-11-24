import streamlit as st
import datetime
import numpy as np
import pandas as pd
import pandas_datareader.data as dat
import yfinance as yf
# from functions import model_pred, plot
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error as mse

#funtion to plot the plots :D :
def plot(aap,valid,m_name,s_name):
#   fig = plt.figure()
  plt.title("{}'s Stock Price Prediction using {} Model".format(s_name,m_name))
  plt.xlabel("Days", fontsize = 14)
  plt.ylabel("Close Price USD ($)", fontsize = 14)
  plt.plot(aap['Date'],aap["Close"])
  plt.plot(valid['Date'],valid[["Close", "Predictions"]])
  plt.legend(["Original", "Valid", "Predictions"])
  plt.show()

# Fit Model function:
def model_pred(aap, days,m_name):
  st.header('{} Model: '.format(m_name))
  futureDays = days

  # create a new target column shifted 'X' units/days up
  aap["Prediction"] = aap[["Close"]].shift(-futureDays)

  # converting the datetime type to seperate days, months and years.
  aap["month"] = aap["Date"].dt.month
  aap["year"] = aap["Date"].dt.year
  aap["day"] = aap["Date"].dt.day

  # initializing features which are: Open High Low Close AdjClose Volume day month and year
  x = aap.drop(["Prediction","Date"],axis = 1).head(-futureDays)
  y = aap["Prediction"].head(-futureDays)

  # Initializing the linear regression model
  if m_name == 'Linear':
    model = LinearRegression().fit(x, y)
  elif m_name == 'DecisionTree':
    model = DecisionTreeRegressor().fit(x, y)
  else:
    model = RandomForestRegressor(n_estimators=100, random_state=42).fit(x, y)

  # taking the last n inputs to be predicted
  x_tobePred = aap.drop(["Prediction","Date"],axis = 1).tail(futureDays)

  # Predictions:
  Prediction = model.predict(x_tobePred)

  # Finding the Root Mean Squared Error:
  rmse = np.round(np.sqrt(mse(Prediction,aap['Close'].tail(futureDays))),3)
  st.write('RMSE: ', rmse)
  valid = pd.DataFrame({'Date': aap['Date'].tail(days),'Close': aap['Close'].tail(days)})
  valid['Predictions'] = Prediction
  return valid

start_date = datetime.datetime(2019, 1, 1)
end_date = datetime.datetime.now()
stock = st.selectbox('Select a company for which you want to see the price prediction: ',['Apple','Google','Tesla','Meta','Amazon'])
if stock == 'Apple':
    data = yf.download('AAPL', start=start_date, end=end_date)
elif stock == 'Google':
    data = yf.download('GOOGL', start=start_date, end=end_date)
elif stock == 'Tesla':
    data = yf.download('TSLA', start=start_date, end=end_date)
elif stock == 'Meta':
    data = yf.download('META', start=start_date, end=end_date)
elif stock == 'Amazon':
    data = yf.download('AMZN', start=start_date, end=end_date)
    
aap = data.reset_index()

st.header('{} Stock Price Prediction'.format(str(stock)))

aap['MovingAverage14'] = aap['Close'].rolling(window=14).mean()
aap['MovingAverage30'] = aap['Close'].rolling(window=30).mean()
aap.dropna(inplace=True)

opt = st.sidebar.selectbox('Enter you model',['Linear','DecisionTree','RandomForest'])
days = st.sidebar.slider("Select the number of days",1,100,25)
if opt == 'Linear':
    a = model_pred(aap,days,str(opt))
    plot(aap,a,str(opt),str(stock))
elif opt == 'DecisionTree':
    a = model_pred(aap,days,str(opt))
    plot(aap,a,str(opt),str(stock))
else:
    a = model_pred(aap,days,str(opt))
    plot(aap,a,str(opt),str(stock))