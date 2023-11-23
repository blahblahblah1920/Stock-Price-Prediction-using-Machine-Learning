import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error as mse
import pandas as pd
import numpy as np
import streamlit as st

#funtion to plot the plots :D :
def plot(aap,valid,m_name,s_name):
  fig = plt.figure()
  plt.title("{}'s Stock Price Prediction using: {} Model".format(s_name,m_name))
  plt.xlabel("Days")
  plt.ylabel("Close Price USD ($)")
  plt.plot(aap['Date'],aap["Close"])
  plt.plot(valid['Date'],valid[["Close", "Predictions"]])
  plt.legend(["Original", "Valid", "Predictions"])
  st.plotly_chart(fig)

# Linear model function:
def linear_reg(aap, days):
  st.header('Liner Model: ')
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
  linear = LinearRegression().fit(x, y)

  # taking the last n inputs to be predicted
  x_tobePred = aap.drop(["Prediction","Date"],axis = 1).tail(futureDays)

  # Predictions:
  linearPrediction = linear.predict(x_tobePred)

  # Finding the Root Mean Squared Error:
  rmse = np.round(np.sqrt(mse(linearPrediction,aap['Close'].tail(futureDays))),3)
  st.write('RMSE: ', rmse)
  valid = pd.DataFrame({'Date': aap['Date'].tail(days),'Close': aap['Close'].tail(days)})
  valid['Predictions'] = linearPrediction
  return valid

# Decision Tree Model Function:
def decision_tree(aap, days):
  st.header('Decision Tree Model: ')
  futureDays = days

  # create a new target column shifted 'X' units/days up
  aap["Prediction"] = aap[["Close"]].shift(-futureDays)

  # converting the datetime type to seperate days, months and years.
  aap["month"] = aap["Date"].dt.month
  aap["year"] = aap["Date"].dt.year
  aap["day"] = aap["Date"].dt.day

  # initializing features which are: Open High Low Close AdjClose Volume day month and year
  x = aap.drop(["Prediction","Date"],axis = 1).head(-futureDays)   #  aap.drop(['Prediction'],axis = 1)
  y = aap["Prediction"].head(-futureDays)

  # Initializing the Decision tree model
  tree = DecisionTreeRegressor().fit(x, y)

  # taking the last n inputs to be predicted
  x_tobePred = aap.drop(["Prediction","Date"],axis = 1).tail(futureDays)

  # Predictions:
  treePrediction = tree.predict(x_tobePred)

  # Finding the Root Mean Squared Error:
  rmse = np.round(np.sqrt(mse(treePrediction,aap['Close'].tail(futureDays))),3)
  st.write('RMSE: ', rmse)
  valid = pd.DataFrame({'Date': aap['Date'].tail(days),'Close': aap['Close'].tail(days)})
  valid['Predictions'] = treePrediction
  return valid

# Random Forest Model Function:  
def random_forest(aap, days):
  st.header('Random Forest Model: ')
  futureDays = days

  # create a new target column shifted 'X' units/days up
  aap["Prediction"] = aap[["Close"]].shift(-futureDays)

  # converting the datetime type to seperate days, months and years.
  aap["month"] = aap["Date"].dt.month
  aap["year"] = aap["Date"].dt.year
  aap["day"] = aap["Date"].dt.day

  # initializing features which are: Open High Low Close AdjClose Volume day month and year
  x = aap.drop(["Prediction","Date"],axis = 1).head(-futureDays)   #  aap.drop(['Prediction'],axis = 1)
  y = aap["Prediction"].head(-futureDays)

  # Initializing the Decision tree model
  rand = RandomForestRegressor(n_estimators=100, random_state=42).fit(x, y)

  # taking the last n inputs to be predicted
  x_tobePred = aap.drop(["Prediction","Date"],axis = 1).tail(futureDays)

  # Predictions:
  randPrediction = rand.predict(x_tobePred)

  # Finding the Root Mean Squared Error:
  rmse = np.round(np.sqrt(mse(randPrediction,aap['Close'].tail(futureDays))),3)
  st.write('RMSE: ', rmse)
  valid = pd.DataFrame({'Date': aap['Date'].tail(days),'Close': aap['Close'].tail(days)})
  valid['Predictions'] = randPrediction
  return valid