import streamlit as st
import joblib
import pandas as pd
import pickle

st.set_page_config(page_title="Churn Prediction")

# Load the pre-trained model
with open('df.pkl', 'rb') as file:
    df = pickle.load(file)

# Load the pipeline using joblib
pipeline = joblib.load('pipeline.joblib')

# Streamlit app
st.title("Customer Churn Prediction")

# Create input fields for each feature in the dataframe
st.header('Customer Data Input')
input_data = {}
for column in df.columns:
    if df[column].dtype == 'object':
        input_data[column] = st.text_input(f"Enter {column}")
    elif df[column].dtype in ['int64', 'float64']:
        input_data[column] = st.number_input(f"Enter {column}", value=float(df[column].median()))

# Convert input data to dataframe
input_df = pd.DataFrame([input_data])

# Predict churn
if st.button("Predict"):
    prediction = pipeline.predict(input_df)
    st.write("Churn Prediction:", "Yes" if prediction[0] == 1 else "No")

# Optionally display the input data for verification
st.write("Input Data")
st.write(input_df)
