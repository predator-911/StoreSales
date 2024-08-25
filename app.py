import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor  # Ensure this is the CPU version
import joblib

# Load the model and encoders (if needed)
@st.cache_resource
def load_model():
    return joblib.load('xgb_model_current_version.pkl')  # Updated model filename

model = load_model()

# Streamlit app UI
st.title('Store Sales Prediction App')
st.write('This app predicts store sales using a pre-trained XGBoost model.')

# User input
st.subheader('Input Features')

# Input fields for user
store_nbr = st.number_input('Store Number', min_value=1, max_value=50, value=1)
family = st.selectbox('Family', ['FOOD', 'HOBBIES', 'HOME', 'ELECTRONICS', 'CLOTHING'])
city = st.text_input('City')
state = st.text_input('State')
type_ = st.selectbox('Store Type', ['A', 'B', 'C'])  # Ensure 'type' exists in your data
locale = st.selectbox('Locale', ['CITY', 'STATE'])
locale_name = st.text_input('Locale Name')
date = st.date_input('Date')

# Load additional data for encoding and merging
stores = pd.read_csv('stores.csv')
oil = pd.read_csv('oil.csv')
holidays_events = pd.read_csv('holidays_events.csv')

# Convert 'date' columns to datetime
oil['date'] = pd.to_datetime(oil['date'])
holidays_events['date'] = pd.to_datetime(holidays_events['date'])

# Prepare the input data
input_data = pd.DataFrame({
    'store_nbr': [store_nbr],
    'family': [family],
    'city': [city],
    'state': [state],
    'type': [type_],  # Ensure 'type' exists in your data
    'locale': [locale],
    'locale_name': [locale_name],
    'date': [pd.to_datetime(date)]  # Convert user input date to datetime
})

# Merge data
input_data = pd.merge(input_data, stores, on='store_nbr', how='left')
input_data = pd.merge(input_data, oil, on='date', how='left')
input_data['dcoilwtico'].ffill(inplace=True)  # Updated fillna method

# Rename columns to match the model's expected feature names
input_data.rename(columns={
    'type': 'type_x',
    'city': 'city_x',
    'state': 'state_x',
    'locale': 'locale_x',
    'locale_name': 'locale_name_x'
}, inplace=True)

# Drop any columns not expected by the model
input_data = input_data[[
    'store_nbr', 'family', 'city_x', 'state_x', 'type_x', 'locale_x',
    'locale_name_x', 'cluster', 'dcoilwtico', 'year', 'month', 'day', 
    'dayofweek', 'is_weekend', 'is_holiday'
]]

# Encode categorical features
le = LabelEncoder()
categorical_cols = ['family', 'locale_x', 'locale_name_x']
for col in categorical_cols:
    input_data[col] = le.fit_transform(input_data[col])

# Create date features
def create_date_features(df):
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['dayofweek'] = df['date'].dt.dayofweek
    df['is_weekend'] = df['dayofweek'] >= 5
    return df

input_data = create_date_features(input_data)

# Incorporate holidays and events information
holidays_events = holidays_events[holidays_events['transferred'] == 'False']
holidays_events = holidays_events[['date', 'type', 'locale', 'locale_name']]
holidays_events['is_holiday'] = 1

input_data = pd.merge(input_data, holidays_events, on=['date'], how='left')
input_data['is_holiday'].fillna(0, inplace=True)

# Ensure all columns are numerical
input_data = input_data.apply(pd.to_numeric, errors='coerce')

# Prepare the data for prediction
X_input = input_data.drop(['date'], axis=1, errors='ignore')

# Predict using the model
if st.button('Predict Sales'):
    prediction = model.predict(X_input)[0]
    st.write(f'Predicted Sales: {prediction}')

    # Prepare the results for download
    results = pd.DataFrame({
        'Store Number': [store_nbr],
        'Date': [date],
        'Predicted Sales': [prediction]
    })
    
    # Provide a download button for the results
    csv = results.to_csv(index=False).encode('utf-8')
    st.download_button("Download Prediction as CSV", csv, "sales_prediction.csv", "text/csv")

# Option to upload a CSV file for batch prediction
st.subheader('Batch Prediction')

uploaded_file = st.file_uploader("Upload CSV file with your data for batch prediction", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    
    # Process the uploaded file similar to the user input
    if 'date' in data.columns:
        data['date'] = pd.to_datetime(data['date'])  # Convert uploaded dates to datetime
        data = pd.merge(data, stores, on='store_nbr', how='left')
        data = pd.merge(data, oil, on='date', how='left')
        data['dcoilwtico'].ffill(inplace=True)

        data.rename(columns={
            'type': 'type_x',
            'city': 'city_x',
            'state': 'state_x',
            'locale': 'locale_x',
            'locale_name': 'locale_name_x'
        }, inplace=True)

        data = create_date_features(data)
        data = pd.merge(data, holidays_events, on=['date'], how='left')
        data['is_holiday'].fillna(0, inplace=True)
        data = data.apply(pd.to_numeric, errors='coerce')
        
        X_batch = data.drop(['date'], axis=1, errors='ignore')
        predictions_batch = model.predict(X_batch)

        results_batch = pd.DataFrame({
            'Index': data.index,
            'Predicted Sales': predictions_batch
        })

        st.write('Batch Predictions:')
        st.write(results_batch)

        # Option to download the batch predictions
        csv_batch = results_batch.to_csv(index=False).encode('utf-8')
        st.download_button("Download Batch Predictions as CSV", csv_batch, "batch_predictions.csv", "text/csv")
    else:
        st.warning("'date' column is missing in the uploaded file.")
