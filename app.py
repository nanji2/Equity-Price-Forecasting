import streamlit as st
import pandas as pd
import yfinance as yf

from datetime import date
from plotly import graph_objs as go
from prophet import Prophet
from pmdarima import auto_arima


@st.cache_resource
def scrape_sp500():
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    df = pd.read_html(url)[0]
    return df


def retrieve_tickers():
    df = scrape_sp500()
    return df['Symbol'].tolist()


def retrieve_stock_info(ticker):
    df = scrape_sp500()
    return df[df['Symbol'] == ticker]


def load_data(stock, start_date, today_date):
    stock_data = yf.download(stock, start_date, today_date)
    stock_data.reset_index(inplace=True)
    return stock_data


def visualise(data):
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name='Open'))
    fig1.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='Close'))
    fig1.layout.update(xaxis_title='Date', yaxis_title='Price')
    fig1.layout.update(xaxis_rangeslider_visible=True)
    st.plotly_chart(fig1)


def predict(model, data, period):
    if model == 'Prophet':
        df_train = data[['Date', 'Close']]
        df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})
        model = Prophet()
        model.fit(df_train)
        future = model.make_future_dataframe(periods=period)
        forecast = model.predict(future)
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='Actual'))
        fig2.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name='Forecast'))
        fig2.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], fill='tonexty',
                                  fillcolor='rgba(68, 68, 68, 0.3)', mode='none', name='Lower Bound', showlegend=False))
        fig2.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], fill='tonexty',
                                  fillcolor='rgba(68, 68, 68, 0.3)', mode='none', name='Upper Bound', showlegend=False))
        fig2.layout.update(xaxis_title='Date', yaxis_title='Price')
        fig2.layout.update(xaxis_rangeslider_visible=True)
        st.plotly_chart(fig2)
    elif model == 'ARIMA':
        df_train = data['Close']
        st.write('Searching for optimal parameters... This may take a while...')
        model = auto_arima(df_train, start_p=0, start_q=0, max_p=5, max_q=5, m=7, start_P=0, seasonal=True,
                           d=1, D=1, trace=True, error_action='ignore', suppress_warnings=True, stepwise=True)
        st.write('Optimal parameters found!')
        st.write(model)
        model_fit = model.fit(df_train)
        forecast = model_fit.predict(period)
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='Close'))
        fig2.add_trace(go.Scatter(x=pd.date_range(start=data['Date'].iloc[-1], periods=period+1, freq='D')[1:],
                                  y=forecast, name='Forecast'))
        fig2.layout.update(xaxis_title='Date', yaxis_title='Price')
        fig2.layout.update(xaxis_rangeslider_visible=True)
        st.plotly_chart(fig2)


def main():
    st.title("S&P500 Stock Price Prediction")
    st.sidebar.title('User Input')

    stocks = retrieve_tickers()
    selected_stock = st.sidebar.selectbox('Select stock:', stocks)
    year = st.sidebar.slider('Select start year:', 2010, 2023)
    start = str(year) + '-01-01'
    today = date.today().strftime('%Y-%m-%d')
    n_years = st.sidebar.slider('Select number of years to forecast:', 1, 5)
    period = n_years * 365

    # Show stock data
    stock_info = retrieve_stock_info(selected_stock)
    st.write('Selected Stock: ' + stock_info['Security'].tolist()[0])
    st.subheader('Show recent data')
    data = load_data(selected_stock, start, today)
    st.write(data.tail(10))

    # Plot stock data
    st.subheader('Visualise trend')
    visualise(data)

    # Select model
    st.subheader('Forecast data')
    selected_model = st.sidebar.radio('Select model:', ['Prophet', 'ARIMA'])
    predict(selected_model, data, period)


if __name__ == '__main__':
    main()
    