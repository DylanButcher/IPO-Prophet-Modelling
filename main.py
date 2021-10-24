import streamlit as st
from datetime import date
import yfinance as yf
from fbprophet import Prophet
from fbprophet.plot import plot_plotly
from plotly import graph_objs as go
import csv

st.title("Stock prediction app")


def getTickersFromTxt():
    with open("stocksList.csv", newline="")as csvfile:
        reader = csv.reader(csvfile, delimiter=" ", quotechar="|")
        for row in reader:
            if row == "":
                continue
            tickerList.append(row)
    return tickerList

def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_Open"))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_Close"))
    fig.layout.update(title_text="Time Series Data", xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

tickerList = []
stocks = getTickersFromTxt()
stock_to_forecast = st.selectbox("Select ticker to forecast", stocks)

START = ("2020-01-01")
TODAY = date.today().strftime("%Y-%m-%d")

months = st.slider("Months of prediction:", 1, 36)
time_to_forecast = months * 31


@st.cache
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

data = load_data(stock_to_forecast)

st.subheader("Raw data")
st.write(data.tail())

plot_raw_data()

#format for prophet
df_train = data[['Date', 'Close']]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

#forecast prophet
m = Prophet(seasonality.mode = 'multiplicative')
m.fit(df_train)
future = m.make_future_dataframe(periods=time_to_forecast)
forecast = m.predict(future)

#display forecast
st.subheader("Forecasted data")
st.write(forecast.tail())

st.write("Forecasted data")
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)
