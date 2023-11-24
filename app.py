import streamlit as st
import datetime
import numpy as np
import pandas as pd
import pandas_datareader.data as dat
from sklearn.metrics import mean_squared_error as mse
import yfinance as yf
from functions import model_pred, plot

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