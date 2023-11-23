import streamlit as st
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader.data as dat
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error as mse
import datetime
import yfinance as yf
from functions import linear_reg, decision_tree, random_forest, plot

start_date = datetime.datetime(2019, 1, 1)
end_date = datetime.datetime.now()
data = yf.download('AAPL', start=start_date, end=end_date)

data.to_csv('Apple2010.csv')

aap = pd.read_csv(r'C:\Users\prana\OneDrive\Desktop\Streamlit\Apple2010.csv')

aap['Date'] = pd.to_datetime(aap['Date'])

opt = st.sidebar.selectbox('Enter you model',['Linear','DecisionTree','RandomForest'])
days = st.sidebar.slider("Select the number of days",1,100,25)
if opt == 'Linear':
    a = linear_reg(aap,days)
    plot(aap,a,str(opt))
elif opt == 'DecisionTree':
    b = decision_tree(aap,days)
    plot(aap,b,str(opt))
else:
    c = random_forest(aap,days)
    plot(aap,c,str(opt))