import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error as mse
import pandas as pd
import numpy as np
import streamlit as st

def plot(aap,valid,m_name):
  fig = plt.figure()
  plt.title("Apple's Stock Price Prediction using: {} Model".format(m_name))
#   plt.xlabel("Days")
#   plt.ylabel("Close Price USD ($)")
#   #plt.subplot(2,2,1)
  plt.plot(aap['Date'],aap["Close"])
  plt.plot(valid['Date'],valid[["Close", "Predictions"]])
  plt.legend(["Original", "Valid", "Predictions"])
  st.plotly_chart(fig)
  #plt.legend(["Original", "Valid", "Predictions"])
  #plt.subplot(2,2,3)
#   plt.plot(aap["Close"])
#   plt.plot(valid[["Close", "Predictions"]])
#   plt.xlim(1100, len(aap['Close']))
#   plt.title("Apple's Stock Price Prediction using Linear Model")
#   plt.xlabel("Days")
#   plt.ylabel("Close Price USD ($)")
#   plt.legend(['original',"Valid", "Predictions"])
  #plt.xticks(aap['Date'])
#   st.plotly_chart(fig)

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
  rmse = np.sqrt(mse(linearPrediction,aap['Close'].tail(futureDays)))
  st.write('RMSE: ', rmse)
  valid = pd.DataFrame({'Date': aap['Date'].tail(days),'Close': aap['Close'].tail(days)})
  valid['Predictions'] = linearPrediction
  return valid
  
  
# Decision Tree Model Function:
def decision_tree(aap, days):
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
  rmse = np.sqrt(mse(treePrediction,aap['Close'].tail(futureDays)))
  st.write('RMSE: ', rmse)
  valid = pd.DataFrame({'Date': aap['Date'].tail(days),'Close': aap['Close'].tail(days)})
  valid['Predictions'] = treePrediction
  return valid


# Random Forest Model Function:  
def random_forest(aap, days):
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
  rmse = np.sqrt(mse(randPrediction,aap['Close'].tail(futureDays)))
  st.write('RMSE: ', rmse)
  valid = pd.DataFrame({'Date': aap['Date'].tail(days),'Close': aap['Close'].tail(days)})
  valid['Predictions'] = randPrediction
  return valid