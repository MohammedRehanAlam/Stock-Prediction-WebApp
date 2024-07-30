# pip install streamlit   or   pip uninstall streamlit   and   pip install streamlit yfinance prophet plotly
# pip install yfinance
# pip install prophet
# pip install plotly
# pip install pandas-datareader
# streamlit run stock_prediction_app.py

import streamlit as st # for building and displaying the interactive web application
from datetime import date # for manipulating date values in the code

import yfinance # for downloading historical stock price data
from prophet import Prophet # for building and fitting the forecasting model
from prophet.plot import plot_plotly # for visualizing the forecast results
from plotly import graph_objs as go # for creating interactive charts and graphs
import pandas_datareader as pdr # alternative data source
import pandas as pd # for data manipulation

#--------------------------------------------------------------------#

st.title("Stock prediction app")

st.markdown(
  """
  <hr>
  <header>
    <p style="text-align: center; font-size: 40px; font-style: bolder">BY</p>
    <p style="text-align: center">MOHAMMED REHAN ALAM-------KANTHETI BHARAT KUMAR-------PANNALA VARUN KUMAR REDDY</p>
    <p style="text-align: center">.............21P61A66B3..................................21P61A6677....................................21P61A66D4..............</p>
  </header>
  <hr>
  """, unsafe_allow_html=True)

#--------------------------------------------------------------------#

stock_names = [
    "Google --> GOOG", "Apple --> AAPL", "Microsoft --> MSFT", 
    "GameStop Corp --> GME", "Gold --> GC=F", "Tesla --> TSLA", 
    "Crude Oil --> CL=F", "Bitcoin_USD --> BTC-USD", "USD/INR --> INR=X",
    "One97 Communications Limited --> PAYTM.NS"
]

# Generate text content
text_content = "\n".join([f"{i + 1}. {name}" for i, name in enumerate(stock_names)])

# Display text box with scrollability
st.subheader("Available Stocks:")
with st.expander("Click to expand", expanded=False):
    st.text_area("", text_content, height=250)

# Define stocks and their corresponding numbers for selection
stocks = ("GOOG", "AAPL", "MSFT", "GME", "GC=F", "TSLA", "CL=F", "BTC-USD", "INR=X", "PAYTM.NS") 
selected_stock = st.selectbox("Select stock", stocks)

# Display selected stock information
selected_stock_index = stocks.index(selected_stock)
selected_stock_name = stock_names[selected_stock_index]
st.write(f"Selected stock: {selected_stock_name}")

#--------------------------------------------------------------------#

# Download stock data and load it
START = "2016-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

n_years = st.slider("Years of prediction:", 1, 2)
period = n_years * 365

@st.cache_data
def load_data(ticker):
    try:
        # Try yfinance first
        data = yfinance.download(ticker, START, TODAY)
        if len(data) < 2:
            raise Exception("Not enough data from yfinance")
        data.reset_index(inplace=True)
        return data
    except Exception as e:
        st.warning(f"Error downloading data from yfinance: {str(e)}. Trying alternative source...")
        try:
            # If yfinance fails, try pandas_datareader
            data = pdr.get_data_yahoo(ticker, start=START, end=TODAY)
            if len(data) < 2:
                raise Exception("Not enough data from pandas_datareader")
            data.reset_index(inplace=True)
            return data
        except Exception as e2:
            st.error(f"Error downloading data from alternative source: {str(e2)}")
            return None

data = load_data(selected_stock)

if data is not None and not data.empty:
    st.subheader("Stock data")
    st.write(data.tail())

    #--------------------------------------------------------------------#
    # def plot_raw_data():
    # Plot stock data
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=data["Date"], y=data["Open"], name="stock_open"))
    fig1.add_trace(go.Scatter(x=data["Date"], y=data["Close"], name="stock_close"))
    fig1.layout.update(title_text="Time Series data", xaxis_rangeslider_visible=True)
    st.plotly_chart(fig1)

    # plot_raw_data()
    #--------------------------------------------------------------------#

    # Predict future prices
    df_train = data[["Date", "Close"]]
    df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

    try:
        m = Prophet()
        m.fit(df_train)
        future = m.make_future_dataframe(periods=period)
        forecast = m.predict(future)

        #--------------------------------------------------------------------#

        # Plot forecast
        st.subheader("Forecast data")
        fig2 = plot_plotly(m, forecast)
        st.plotly_chart(fig2)
    except Exception as e:
        st.error(f"Error in prediction: {str(e)}")

else:
    st.warning("No data available for the selected stock. Please try another stock or check your internet connection.")

#--------------------------------------------------------------------#

# Footer with about and contact information
st.markdown(
  """
  <hr>
  <footer>
    <p style="text-align: center; font-size: 40px; font-style: bolder">THANK YOU</p>
    <p>About: This application is developed by Mohammed Rehan Alam as a tool for stock price prediction. It utilizes Prophet library for forecasting and Streamlit for building the web interface.</p>
    <p>Contact: For any questions or feedback, please reach out to mohammedrehanalam@email.com</p>
  </footer>
  <hr>
  """, unsafe_allow_html=True)