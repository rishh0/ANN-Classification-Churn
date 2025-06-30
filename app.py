## End to End Streamlit App for Customer Churn Prediction

import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pickle

# Load the trained model and scaler
model = tf.keras.models.load_model('model.h5')

# Load the encoders and scalers
with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('one_hot_encoder_geo.pkl', 'rb') as file:
    label_encoder_geo = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)


# StreamLit App
st.title("Customer Churn Prediction")

# User Input
geography = st.selectbox("Geography", label_encoder_geo.categories_[0])
gender = st.selectbox("Gender", label_encoder_gender.classes_)
age = st.slider("Age", 18, 92)
balance = st.number_input("Balance")
credit_Score = st.number_input("Credit Score")
estimated_salary = st.number_input("Estimated Salary")
tenure = st.slider("Tenure", 0, 10)
num_of_products = st.slider("Number of Products", 1, 4)
has_cr_card = st.selectbox("Has Credit Card", [1, 0])
is_active_member = st.selectbox("Is Active Member", [1, 0])

# Prepare the input data
input_data = pd.DataFrame({
    'CreditScore' : [credit_Score],
    'Gender' : [label_encoder_gender.transform([gender])[0]],
    'Age' : [age],
    'Tenure' : [tenure],
    'Balance' : [balance],
    'NumOfProducts' : [num_of_products],
    'HasCrCard' : [has_cr_card],
    'IsActiveMember' : [is_active_member],
    'EstimatedSalary' : [estimated_salary]
})

# One-hot encode the Geography
geo_encoded = label_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=label_encoder_geo.get_feature_names_out(['Geography']))

# Concatenate the one-hot encoded Geography with the input data
input_data = pd.concat([input_data.reset_index(drop = True), geo_encoded_df], axis=1)

# Rename columns to match the feature names used during scaler fitting
input_data.rename(columns={
	'Credit Score': 'CreditScore',
	'Estimated Salary': 'EstimatedSalary',
	'IsActiveCustomer': 'IsActiveMember'
}, inplace=True)

# Scaling the Data
input_scaled = scaler.transform(input_data)

# Predict Churn
prediction = model.predict(input_scaled)
prediction_proba = prediction[0][0]

if prediction_proba > 0.5:
    st.success(f"The customer is likely to churn with a probability of {prediction_proba * 100:.2f}%")
else:
    st.success(f"The customer is likely to stay with a probability of {(1 - prediction_proba) * 100:.2f}%")