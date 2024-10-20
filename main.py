import streamlit as st
from datetime import datetime

import yfinance as yf 
from fbprophet import Prophet
from fbprophet.plot import plot_plotly
from plotly import graph_objs as go 



st.title("Stock Prediction App")

stocks = ("AAPL","GOOG","MSFT","GME")
selected_stocks=st.selectbox("Select dataset for prediction")


