import streamlit as st
from datetime import date
import yfinance as yf 
from prophet import Prophet
from plotly import graph_objs as go 

START = "2017-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title("Stock Prediction App")
stocks = ('^NSEI', '^NSEBANK', '^BSESN')
selected_stock = st.selectbox("Select dataset for prediction", stocks)

n_years = st.slider("Years of prediction:", 1, 7)
period = n_years * 365

@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

data_load_state = st.text("Loading data.....")
data = load_data(selected_stock)
data_load_state.text("Loading data.....done!")

st.subheader('Raw data')
st.write(data.tail())

def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name='Stock Open',
                             line=dict(color='blue')))  
    
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='Stock Close',
                             line=dict(color='red'))) 
    
    fig.layout.update(title_text="Time Series Data", xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

plot_raw_data()

# Forecasting

# Select the 'Date' and 'Close' columns
df_train = data[['Date', 'Close']]

# Rename the columns
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

m = Prophet()
m.fit(df_train)

future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

st.subheader('Forecast Data')
st.write(forecast.tail())

# Create a custom figure for the forecast
def plot_forecast(forecast):
    fig = go.Figure()
    
    # Plot the actual close prices
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='Actual Close',
                             line=dict(color='red')))  
    
    # Plot the forecasted values
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name='Forecast',
                             line=dict(color='blue')))  
    
    # Plot the uncertainty intervals
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], name='Upper Bound',
                             line=dict(color='lightblue'), fill='tonexty', fillcolor='rgba(0,0,255,0.2)')) 
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], name='Lower Bound',
                             line=dict(color='lightblue'), fill='tonexty', fillcolor='rgba(255,0,0,0.2)'))  

    fig.layout.update(title_text="Forecast vs Actual", xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

plot_forecast(forecast)

# Display forecast components
st.write('Forecast Components')
fig2 = m.plot_components(forecast)
st.pyplot(fig2)  
