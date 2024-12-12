# all the required modules to be installed.
# pip install streamlit        or     pip install streamlit yfinance prophet plotly pandas-datareader
# pip install yfinance
# pip install prophet
# pip install plotly

# in the end we will use this code in the terminal "streamlit run stock_prediction_app.py"

import streamlit as st                  # for building and displaying the interactive web application
from datetime import datetime, timedelta# for manipulating date values in the code

import yfinance as yf                   # for downloading historical stock price data
from prophet import Prophet             # for building and fitting the model
import plotly.graph_objs as plt
from prophet.plot import plot_plotly    # for visualizing the forecast results
import pandas_datareader as pdr         # alternative data source
import pandas as pd                     # for data manipulation


class User:
    def __init__(self, username=None, email=None):
        """
        Initialize a User with optional username and email.
        
        Args:
            username (str, optional): User's username. Defaults to None.
            email (str, optional): User's email. Defaults to None.
        """
        self.username = username
        self.email = email
        self.favorite_stocks = []
    
    def save_favorite_stocks(self, stocks):
        """
        Save a list of favorite stocks for the user.
        
        Args:
            stocks (List[str]): List of stock symbols to save.
        """
        self.favorite_stocks = list(set(stocks))
    
    def get_favorite_stocks(self):
        """
        Retrieve user's favorite stocks.
        
        Returns:
            List[str]: List of saved stock symbols.
        """
        return self.favorite_stocks

class StockPredictionApp:
    def __init__(self):
        """
        Initialize the Stock Prediction Streamlit App.
        """
        # Predefined stock examples with full names and symbols
        self.stock_examples = [
            {"name": "Google", "symbol": "GOOG"},
            {"name": "Apple", "symbol": "AAPL"},
            {"name": "Microsoft", "symbol": "MSFT"},
            {"name": "Tesla", "symbol": "TSLA"},
            {"name": "Amazon", "symbol": "AMZN"},
            {"name": "NVIDIA", "symbol": "NVDA"},
            {"name": "Meta-Facebook", "symbol": "META"},
            {"name": "Bitcoin USD", "symbol": "BTC-USD"},
            {"name": "Gold", "symbol": "GC=F"}
        ]
    
    def validate_stock_symbol(self, symbol):
        """
        Validate if the given stock symbol exists.
        
        Args:
            symbol (str): Stock symbol to validate.
        
        Returns:
            bool: True if symbol is valid, False otherwise.
        """
        try:
            stock = yf.Ticker(symbol)
            # Try to fetch basic info to validate
            stock.info
            return True
        except Exception:
            return False
    
    def fetch_stock_data(self, symbol, start_date, end_date):
        """
        Fetch stock data using yfinance.
        
        Args:
            symbol (str): Stock symbol to fetch.
            start_date (datetime): Start date for data retrieval.
            end_date (datetime): End date for data retrieval.
        
        Returns:
            pd.DataFrame: Stock price data.
        """
        try:
            stock_data = yf.download(symbol, start=start_date, end=end_date)
            return stock_data
        except Exception as e:
            st.error(f"Error fetching data for {symbol}: {e}")
            return pd.DataFrame()
    
    def run(self):
        """
        Main method to run the Streamlit application.
        """
        
        # Set page title and favicon
        st.set_page_config(page_title="Stock Prediction App", page_icon=":chart_with_upwards_trend:")
        
        # App title
        st.title("📈 Stock Prediction App")

        st.markdown(
        """
        <hr>
        <header>
            <p style="text-align: center; font-size: 40px; font-style: bolder">BY</p>
            <p style="text-align: center">MOHAMMED REHAN ALAM-------KANTHETI BHARAT KUMAR-------PANNALA VARUN KUMAR REDDY</p>
            <p style="text-align: center">.............21******B3..................................21******77....................................21******D4..............</p>
        </header>
        <hr>
        """, unsafe_allow_html=True)

        # Create two columns for stock selection
        col1, col2 = st.columns(2)
        
        # Dropdown for predefined stocks
        with col1:
            st.subheader("Select from Examples")
            # Create list of example stock names for dropdown
            example_stock_options = [f"{stock['name']} ({stock['symbol']})" for stock in self.stock_examples]
            selected_example = st.selectbox("Choose a stock", example_stock_options)
            
            # Extract symbol from selection
            selected_symbol = selected_example.split('(')[1].strip(')')
        
        # Manual stock symbol input
        with col2:
            st.subheader("Or Search Any Stock")
            manual_symbol = st.text_input("Enter Stock Symbol", placeholder="e.g., AAPL, GOOG")
            
            # Use manual input if provided, otherwise use example
            stock_symbol = manual_symbol.upper() if manual_symbol else selected_symbol
        
        # Display the selected stock symbol
        st.write(f"Selected Stock: **{stock_symbol}**")
        
        # Validate stock symbol
        with st.expander("Check Stock Information"):
            if not self.validate_stock_symbol(stock_symbol):
                st.error(f"Valid stock symbol: {stock_symbol}")
                return


        # Date range and prediction settings
        START = "2010-01-01" 
        # START = (datetime.now() - timedelta(days=3*365)).strftime("%Y-%m-%d")
        TODAY = datetime.now().strftime("%Y-%m-%d")
        
        # Prediction years slider
        n_years = st.slider("Years of prediction:", 1, 3, 2)
        period = n_years * 365
        
        # Fetch stock data
        data = self.fetch_stock_data(stock_symbol, START, TODAY)
        
        if data is not None and not data.empty:
            
                            # ---------------------------------------------------------
            # Display recent data
            st.subheader("Recent Stock Data")
            st.write(data.tail())
            
            # Plot historical stock prices
            fig_historical = plt.Figure()
            fig_historical.add_trace(
                plt.Scatter(x=data.index, y=data['Close'], mode='lines', name='Close Price')
            )
            fig_historical.update_layout(
                title=f'{stock_symbol} Historical Stock Prices',
                xaxis_title='Date',
                yaxis_title='Price (USD)'
            )
            st.plotly_chart(fig_historical, use_container_width=True)
                            # -------------------------------------------------------------

            # Prepare data for Prophet
            df_train = pd.DataFrame({
                'ds': data.index.astype(str).values.ravel(),
                'y': data['Close'].values.ravel()
            })
            
            # Predict future prices
            try:
                model = Prophet()
                model.fit(df_train)
                
                future = model.make_future_dataframe(periods=period)
                forecast = model.predict(future)
                
                            # -------------------------------------------------------------
                # Plot forecast alongside original price
                st.subheader(f"{stock_symbol} Price Forecast with Historical Prices")
                forecast_fig = plt.Figure()
                
                # Add historical prices
                forecast_fig.add_trace(
                    plt.Scatter(
                        x=data.index.astype(str),
                        y=data['Close'].values,
                        mode='lines',
                        name='Historical Prices'
                    )
                )
                
                # Add forecast line
                forecast_fig.add_trace(
                    plt.Scatter(
                        x=forecast['ds'].astype(str),
                        y=forecast['yhat'].values,
                        mode='lines',
                        name='Forecast',
                        line=dict(color='blue')
                    )
                )
                
                # Add upper and lower bounds
                forecast_fig.add_trace(
                    plt.Scatter(
                        x=forecast['ds'].astype(str),
                        y=forecast['yhat_upper'].values,
                        mode='lines',
                        name='Upper Bound',
                        line=dict(width=0, color='rgba(0,0,0,0)'),
                        showlegend=True
                    )
                )
                forecast_fig.add_trace(
                    plt.Scatter(
                        x=forecast['ds'].astype(str),
                        y=forecast['yhat_lower'].values,
                        mode='lines',
                        name='Lower Bound',
                        line=dict(width=0, color='rgba(0,0,0,0)'),
                        fill='tonexty',
                        fillcolor='rgba(68, 68, 68, 0.3)',
                        showlegend=True
                    )
                )
                
                forecast_fig.update_layout(
                    title=f'{stock_symbol} Price Forecast with Historical Prices',
                    xaxis_title='Date',
                    yaxis_title='Price (USD)'
                )
                
                st.plotly_chart(forecast_fig)
                            # -------------------------------------------------------------
                            
                # Display forecast components
                st.subheader('Forecast Components')
                fig_components = model.plot_components(forecast)
                st.pyplot(fig_components)
                
            except Exception as e:
                st.error(f"Error in prediction: {e}")
        else:
            st.warning("No data available for the selected stock. Please try another stock or check your internet connection.")
        
        # Footer
        st.markdown("---")
        st.markdown(""" 
        ### About the App
        - This Stock Prediction App uses historical stock data to generate future price forecasts.
        - Data sourced from Yahoo Finance using yfinance.
        - Predictions made using Facebook Prophet forecasting model.
        
        **Disclaimer:** Stock predictions are probabilistic and should not be considered financial advice.
        """)

        st.markdown("""
        <hr>
        <footer>
            <p style="text-align: center; font-size: 40px; font-style: bolder">THANK YOU</p>
            <p>About: This application is developed by Mohammed Rehan Alam as a tool for stock price prediction. It utilizes Prophet library for forecasting and Streamlit for building the web interface.</p>
            <p>Contact: For any questions or feedback, please reach out to <a href="mailto:mohammedrehanalam16@email.com" style="text-decoration: none;">mxxxxxxxxxxx@email.com</a></p>
        </footer>
        <hr>
        """, unsafe_allow_html=True)

# Main execution

if __name__ == "__main__":
    app = StockPredictionApp()
    app.run()
