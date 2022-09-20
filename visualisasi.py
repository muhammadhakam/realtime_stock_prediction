#import data

import streamlit as st
from datetime import date
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go

START = "2017-01-02"
TODAY = date.today().strftime("%Y-%m-%d")

#menggunakan web based streamit
st.title("TELKOM Persero (TLKM.J) Forcast Prediction by Muhammad Hakam Amnan_ular")
stock = ("TLKM.JK","SMGR.JK","WIKA.JK")
selected_stock = st.selectbox("Select dataset for prediction", stock)

n_year = st.slider("Year of prediction :", 1,6)
period = n_year * 365

#mengambil data secara real time dari dataset yfinance bisa juga diganti dengan csv file
@st.cache
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

data_load_state = st.text("Loading data...")
data = load_data(selected_stock)
data_load_state.text("Loading data.... done!")

st.subheader("Raw Data")
st.write(data.tail())

#visualisasi data raw dengan index Date dan analisa market Open dan Close
def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open"))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
    fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

plot_raw_data()

#melakukan akuisisi variabel yang digunakan dalam training menggunakan library fbprophet
data_train = data[["Date","Close"]]
data_train = data_train.rename(columns={"Date": "ds", "Close": "y"})

#melakukan training data untuk membuat prediksi berdasarkan waktu (timeseries)
m = Prophet()
m.fit(data_train)
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

st.subheader("Forecast Data")
st.write(forecast.tail())
st.write(f'Forecast plot for {n_year} years')
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)

st.write("Forecast component")
fig2 = m.plot_components(forecast)
st.write(fig2)