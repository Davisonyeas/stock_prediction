
import streamlit as st
from datetime import date

import tkinter
import _tkinter
import turtle
import tk
import tkinter as TK

from PIL import Image
import pandas as pd
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go

im = Image.open("stock.png")
st.set_page_config(
    page_title="Stock Price Prediction by Davis",
    page_icon=im,
    
)

hide_menu = """
<style>
    #MainMenu{
        visibility: hidden;    
    }
    footer {
        visibility: hidden;
    }

</style>
"""

st.markdown(hide_menu, unsafe_allow_html=True)

st.markdown(hide_menu, unsafe_allow_html=True)

START = "2015-01-01"

TODAY = date.today().strftime("%Y-%m-%d")

st.title("Stock Prediction Web App 📈")

st.subheader('Introduction')


st.write('There is an old but famous joke about stocks and the stock market: "If you have lost some money in stock market and feel bad about it, don’t worry. Ask somebody you know about their losses in the stock market, and you will feel better that you lost less money."')

st.write("While this joke has been doing rounds for several decades now and is still quite relevant because there is absolutely no shortage of people who lose money in stock markets daily. According to popular estimates, as much as 90% of people lose money in stock markets, including both new and seasoned investors.")

st.subheader('Description of Project')

st.write('A stock market (aka equity market or share market) is the aggregation of buyers and sellers of stocks, which represent ownership claims on businesses')

st.write('In simple words, stock markets are places where buyers and sellers meet to exchange equity shares of public corporations.')

st.markdown("![Alt Text](https://i.gifer.com/7vpz.gif)")

st.subheader('Aim of Project')

st.write('The aim of this project is to predict or determine the future movement of the stock value of a financial exchange. The accurate prediction of share price movement will lead to more profit investors can make.')

st.write('Full disclosure: I am not a share market king, neither am I a top trader. I am a programmer who knows how to derive insights from historical data. For the purpose of this project, I made research about stock market and I combined various tools to bring this project to life. I ran a predictive analytics of the stock market of some of the most researched companies.')

st.markdown("![Alt Text](https://i.gifer.com/7D7o.gif)")

st.subheader("About the Dataset")

st.write("The 'Symbol' column represents the Stock symbol, they're usually 1 to 4 letter codes that identifies publicly traded companies, closed-end mutual funds, exchange-traded funds, This is the symbol you are to select in the dropdown below (under the Select dataset for prediction).")

st.write("The 'Company Name' column signifies which Company is represented by the Stock symbol , you can make reference to the column from time to time to check the stock prices and predicted stock prices of the Company.")

stock_desc = pd.read_excel(io="stock_desc.xlsx", engine="openpyxl")

st.dataframe(stock_desc)

# stock_desc = st.dataframe()

# st.dataframe()

stocks = ('AAPL', 'GOOG', 'MSFT', 'GME', 'AAPL','MSFT','GOOG','GOOGL','AMZN','META','NVDA','JPM','KO','BABA','DIS','TMUS','AMD','INTC','MMM','OXY','SNOW','F','PXD','WDAY','SHOP','EA','DELL','PLUG','AFRM','SOFI','FTCH','PTON','JWN','MFGP')

selected_stock = st.selectbox("Select dataset for prediction", stocks)

st.write("Increase the range below to predict more years")

n_years = st.slider("Years of prediction", 1, 4)

period = n_years * 365

@st.cache

def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

data_load_state = st.text("Load data...")

data = load_data(selected_stock)

data_load_state.text("Loading data...DONE")


st.subheader("Raw Data")

st.write(data.tail())

def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y = data['Open'], name='stock_open'))

    fig.add_trace(go.Scatter(x=data['Date'], y = data['Close'], name='stock_close'))

    st.write("Hover on the plot below to get the month and year of the stock price, Zoom in to get the particular day")

    fig.layout.update(title_text='Time Series Data', xaxis_rangeslider_visible=True)

    st.plotly_chart(fig)

plot_raw_data()

#Forecast

df_train = data[['Date', 'Close']]

df_train = df_train.rename(columns={'Date': 'ds', 'Close': 'y'})

m = Prophet()

m.fit(df_train)

future = m.make_future_dataframe(periods=period)

forecast = m.predict(future)

st.subheader("Forecast Data")

st.write(forecast.tail())

st.write("Forecast Data")

st.write("The dots represents the day and the actual stock price, the last dot in the series represents previous day stock before you are viewing this")

st.write("NOTE: The data is being gotten from the internet in real-time and will continuously change as days go by, the line is what my algorithm predicts the price will be.")

fig1 = plot_plotly(m, forecast)

st.plotly_chart(fig1)

st.write("Forecast Component")

st.write("This is the prediction, the number of years you selected earlier (the range of 1-4), is the number of years that will be generated and plotted")

fig2 = m.plot_components(forecast)

st.write(fig2)

st.write("Click here to view more of my projects ([Davis Projects)](https://www.davisonye.github.io)")

st.write("Click here to view my portfolio website ([Davis Portfolio)](https://www.davisonye.github.io)")

st.write("Click here to view my digital CV([Davis CV)](https://www.davisonye.github.io)")
