import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error as mse
import pandas as pd
import numpy as np
import streamlit as st
import plotly.tools

#funtion to plot the plots :D :
def plot(aap,valid,m_name,s_name):
  fig = plt.figure()
  plt.title("{}'s Stock Price Prediction using {} Model".format(s_name,m_name))
  plt.xlabel("Days", fontsize = 14)
  plt.ylabel("Close Price USD ($)", fontsize = 14)
  plt.plot(aap['Date'],aap["Close"])
  plt.plot(valid['Date'],valid[["Close", "Predictions"]])
  plt.legend(["Original", "Valid", "Predictions"])
  st.plotly_chart(fig)

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