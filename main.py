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


stocks = getTickersFromTxt()
selected_stock = st.selectbox("Select dataset for prediction", stocks)

START = st.date_input("Start Date")
TODAY = date.today().strftime("%Y-%m-%d")

n_years = st.slider("Years of prediction:", 1, 4)
period = n_years * 365


@st.cache
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

data = load_data(selected_stock)

st.subheader("Raw data")
st.write(data.tail())

plot_raw_data()

# Forecast with prophet
df_train = data[['Date', 'Close']]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

m = Prophet(interval_width=0.95)
m.fit(df_train)
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

st.subheader("Forecasted data")
st.write(forecast.tail())

st.write("Forecasted data")
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)
