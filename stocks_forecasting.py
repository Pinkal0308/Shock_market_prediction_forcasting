import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import cufflinks as cf
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from streamlit_option_menu import option_menu

# Set cufflinks to use plotly
cf.go_offline()
cf.set_config_file(offline=False, world_readable=True)


# Streamlit configuration
st.set_page_config(page_title="Stock Price Forecasting", page_icon=":chart_with_upwards_trend:", layout="wide")

# Set the title in the middle with bold and attractive font
st.markdown(
    """
    <h1 style='text-align: center; font-size: 60px; font-weight: bold; color: white;'>
    Share Market Forecasting And Analysis
    </h1>
    """,
    unsafe_allow_html=True
)


# Streamlit Option Menu for Navigation with improved spacing
selected = option_menu(
    menu_title="Main Menu",
    options=["Home", "Forecasting", "Analysis", "Correlation Matrix", "About"],
    icons=["house", "graph-up-arrow", "bar-chart-line", "diagram-3", "info-circle"],
    menu_icon="cast",
    default_index=0,
    orientation="horizontal"
)

st.write("Upload your CSV file for analysis and forecasting.")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Read CSV
    try:
        data = pd.read_csv(uploaded_file)

        # Automatically detect the date column
        date_column = None
        for col in data.columns:
            if pd.to_datetime(data[col], errors='coerce').notna().all():
                date_column = col
                break
        
        if date_column:
            data[date_column] = pd.to_datetime(data[date_column])
            data.set_index(date_column, inplace=True)
        else:
            st.error("The CSV file must contain a date column.")
            st.stop()

        st.write("Data Preview:")
        st.write(data.head())

        # Handling null values (dynamic one-liners)
        st.write("Handling missing values:")
        data.fillna(method='ffill', inplace=True)
        data.dropna(inplace=True)
        st.write(data.isnull().sum())

        if selected == "Forecasting":
            st.subheader("Forecasting")

            # Select columns for forecasting
            columns = list(data.columns)
            selected_column = st.selectbox("Select the column for forecasting", columns)

            # Plot the historical data with black background
            st.write(f"Plotting historical data for: {selected_column}")
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=data.index, y=data[selected_column],
                mode='lines',
                line=dict(color='rgba(255, 153, 51, 1.0)'),
                name=selected_column
            ))
            fig.update_layout(
                title=f'{selected_column} Over Time',
                xaxis_title='Date',
                yaxis_title=selected_column,
                template='plotly_dark'  # Black background
            )
            st.plotly_chart(fig)

            # Select model parameters
            arima_order = st.text_input("Enter ARIMA order (p,d,q)", "(1,1,1)")
            sarima_order = st.text_input("Enter SARIMA order (p,d,q,P,D,Q,s)", "(1,1,1,1,1,1,12)")

            # Seasonal parameters sliders
            st.subheader('SARIMA Model Forecasting')
            p_s = st.sidebar.slider('p (Seasonal AR term)', 0, 5, 1)
            d_s = st.sidebar.slider('d (Seasonal Difference term)', 0, 2, 1)
            q_s = st.sidebar.slider('q (Seasonal MA term)', 0, 5, 1)
            seasonal_period = st.sidebar.slider('Seasonal Period (days)', 7, 365, 30)

            # ARIMA Model
            try:
                st.write("Building ARIMA model...")
                p, d, q = eval(arima_order)
                arima_model = ARIMA(data[selected_column], order=(p,d,q)).fit()
                arima_forecast = arima_model.forecast(steps=1)
                st.write(f"Next predicted value using ARIMA: {arima_forecast.iloc[0]:.2f}")



                
                # Highlight: Determine whether the next predicted price increases, decreases, or remains constant
                last_known_value = data[selected_column].iloc[-1]
                if arima_forecast.iloc[0] > last_known_value:
                    arima_prediction_result = "The next predicted price will increase."
                elif arima_forecast.iloc[0] < last_known_value:
                    arima_prediction_result = "The next predicted price will decrease."
                else:
                    arima_prediction_result = "The next predicted price will remain constant."
                st.write(arima_prediction_result)  # Display ARIMA prediction result

                
                
                # Display ARIMA model summary
                st.subheader("ARIMA Model Summary")
                st.text(arima_model.summary())
            except Exception as e:
                st.error(f"Error in ARIMA model: {e}")  

            # SARIMA Model
            try:
                st.write("Building SARIMA model...")
                p, d, q, P, D, Q, s = eval(sarima_order)
                sarima_model = SARIMAX(data[selected_column], order=(p,d,q), seasonal_order=(P,D,Q,s)).fit()
                sarima_forecast = sarima_model.forecast(steps=1)
                st.write(f"Next predicted value using SARIMA: {sarima_forecast.iloc[0]:.2f}")



                # Highlight: Determine whether the next predicted price increases, decreases, or remains constant
                if sarima_forecast.iloc[0] > last_known_value:
                    sarima_prediction_result = "The next predicted price will increase."
                elif sarima_forecast.iloc[0] < last_known_value:
                    sarima_prediction_result = "The next predicted price will decrease."
                else:
                    sarima_prediction_result = "The next predicted price will remain constant."
                st.write(sarima_prediction_result)  # Display SARIMA prediction result
                
                # Display SARIMA model summary
                st.subheader("SARIMA Model Summary")
                st.text(sarima_model.summary())
            except Exception as e:
                st.error(f"Error in SARIMA model: {e}")

            # Create a new row with predictions
            next_date = data.index[-1] + pd.Timedelta(days=1)
            new_row = pd.DataFrame({
                selected_column: [None],
                'ARIMA Prediction': [arima_forecast.iloc[0]],
                'SARIMA Prediction': [sarima_forecast.iloc[0]]
            }, index=[next_date])

            # Concatenate the new row to the existing data
            data = pd.concat([data, new_row])

            # Chart with historical data and predictions
            fig_pred = go.Figure()
            fig_pred.add_trace(go.Scatter(x=data.index, y=data[selected_column], mode='lines', name=selected_column))
            fig_pred.add_trace(go.Scatter(x=data.index, y=data['ARIMA Prediction'], mode='lines+markers', name='ARIMA Prediction'))
            fig_pred.add_trace(go.Scatter(x=data.index, y=data['SARIMA Prediction'], mode='lines+markers', name='SARIMA Prediction'))

            fig_pred.update_layout(
                title=f'{selected_column} with ARIMA & SARIMA Predictions',
                xaxis_title='Date',
                yaxis_title=selected_column,
                hovermode='x',
                template='plotly_dark'  # Black background
            )

            st.plotly_chart(fig_pred)

            # Showing the final dataframe with predictions
            st.write("Final Data with Predictions:")
            st.write(data.tail())

        elif selected == "Analysis": # Re-select column for analysis
            columns = list(data.columns)
            selected_column = st.selectbox("Select the column for analysis",columns)

            # Example 1: Moving Average
            st.write("Moving Average (MA) Analysis")
            window_size = st.slider("Select window size for MA", 5, 100, 20)
            data[f'{selected_column} MA ({window_size})'] = data[selected_column].rolling(window=window_size).mean()
    
            fig_ma = go.Figure()
            fig_ma.add_trace(go.Scatter(x=data.index, y=data[selected_column], mode='lines', name=selected_column))
            fig_ma.add_trace(go.Scatter(x=data.index, y=data[f'{selected_column} MA ({window_size})'], mode='lines', name=f'MA ({window_size})'))
            fig_ma.update_layout(
                title=f'{selected_column} with Moving Average (Window={window_size})',
                xaxis_title='Date',
                yaxis_title=selected_column,
                template='plotly_dark'
            )
            st.plotly_chart(fig_ma)

            # Example 2: Bollinger Bands
            st.write("Bollinger Bands Analysis")
            window_size_bb = st.slider("Select window size for Bollinger Bands", 5, 100, 20)
            data['MA'] = data[selected_column].rolling(window=window_size_bb).mean()
            data['BB_up'] = data['MA'] + 2*data[selected_column].rolling(window=window_size_bb).std()
            data['BB_down'] = data['MA'] - 2*data[selected_column].rolling(window=window_size_bb).std()

            fig_bb = go.Figure()
            fig_bb.add_trace(go.Scatter(x=data.index, y=data[selected_column], mode='lines', name=selected_column))
            fig_bb.add_trace(go.Scatter(x=data.index, y=data['BB_up'], mode='lines', name='Upper Bollinger Band', line=dict(color='green')))
            fig_bb.add_trace(go.Scatter(x=data.index, y=data['BB_down'], mode='lines', name='Lower Bollinger Band', line=dict(color='red')))
            fig_bb.update_layout(
                title=f'{selected_column} with Bollinger Bands',
                xaxis_title='Date',
                yaxis_title=selected_column,
                template='plotly_dark'
            )
            st.plotly_chart(fig_bb)

            # Example 3: Volume Analysis (if Volume data is available)
            if 'Volume' in data.columns:
                st.write("Volume Over Time")
                fig_vol = data['Volume'].iplot(asFigure=True, title='Volume Over Time', xTitle='Date', yTitle='Volume')
                st.plotly_chart(fig_vol)

            # Example 4: Return Analysis
            st.write("Daily Return Analysis")
            data['Daily Return'] = data[selected_column].pct_change()
            fig_return = go.Figure()
            fig_return.add_trace(go.Scatter(x=data.index, y=data['Daily Return'], mode='lines', name='Daily Return'))
            fig_return.update_layout(
                title=f'{selected_column} Daily Return',
                xaxis_title='Date',
                yaxis_title='Daily Return',
                template='plotly_dark'
            )
            st.plotly_chart(fig_return)

            # Example 5: Relative Strength Index (RSI)
            st.write("Relative Strength Index (RSI) Analysis")
            window_size_rsi = st.slider("Select window size for RSI", 5, 100, 14)
            delta = data[selected_column].diff()
            gain = (delta.where(delta > 0, 0)).fillna(0)
            loss = (-delta.where(delta < 0, 0)).fillna(0)
            avg_gain = pd.Series(gain).rolling(window=window_size_rsi, min_periods=1).mean()
            avg_loss = pd.Series(loss).rolling(window=window_size_rsi, min_periods=1).mean()
            rs = avg_gain / avg_loss
            data['RSI'] = 100 - (100 / (1 + rs))


            # Check if RSI has valid values to plot
            if data['RSI'].isnull().all():
                st.warning(f"RSI could not be calculated for {selected_column}. Please check the data.")
            else:

                fig_rsi = go.Figure()
                fig_rsi.add_trace(go.Scatter(x=data.index, y=data['RSI'], mode='lines', name='RSI'))
                fig_rsi.update_layout(
                    title=f'{selected_column} Relative Strength Index (RSI)',
                    xaxis_title='Date',
                    yaxis_title='RSI',
                    template='plotly_dark'
                )
                st.plotly_chart(fig_rsi)

        elif selected == "Correlation Matrix":
            st.subheader("Correlation Matrix")
            corr_matrix = data.corr()

            # Create a heatmap
            plt.figure(figsize=(10, 6))
            sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="Blues", square=True, cbar_kws={"shrink": .8})

            # Show the heatmap in Streamlit
            st.pyplot(plt)

        elif selected == "About":
            st.subheader("About This Application")
            st.write("""
                     This Share Market Forecasting application allows users to upload historical stock market data
                     in CSV format and forecast future prices using ARIMA and SARIMA models. Users can visualize
                     historical trends, model predictions, and perform basic statistical analysis on the data.

                    ### Features:
                    - Upload a CSV file with date and stock price information.
                    - Forecast future prices using ARIMA and SARIMA models.
                    - Interactive visualization of historical data and predictions.
                    - Analyze data with descriptive statistics and correlation analysis.

                    ### Technologies Used:
                    - Streamlit for building the web application.
                    - Pandas for data manipulation.
                    - Matplotlib, Seaborn, Plotly, and Cufflinks for interactive visualizations.
                    - Statsmodels for statistical modeling (ARIMA and SARIMA).

                    ### How to Use:
                    1. Upload your CSV file containing stock market data.
                    2. Select the column you want to forecast.
                    3. Enter model parameters for ARIMA and SARIMA.
                    4. View the predictions and visualizations.
                    """)

            st.write("This app provides forecasting and analysis for share market data using ARIMA and SARIMA models.")

    except ValueError as e:
        st.error(f"Error reading the file: {e}")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")

