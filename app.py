import streamlit as st 
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model('model.h5')

## load encoder and scaler
with open('ohe_encoder_geography.pkl', 'rb') as f:
    onehot_encoder_geography = pickle.load(f)

with open('le1_encoder_gender.pkl', 'rb') as f:
    label_encoder_gender = pickle.load(f)

with open('scaler.pkl','rb') as f:
    scaler = pickle.load(f)


# streamlit app
st.title("Bank Customer Churn Prediction")

# user input
geography = st.selectbox('Geography', onehot_encoder_geography.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance', min_value=0.0)
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary', min_value=0.0)
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

# prepare input data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary],
})

# One-hot encode 'Geography'
geo_encoded = onehot_encoder_geography.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geography.get_feature_names_out(['Geography']))

# combine one-hot encoded columns with input data
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

# Scale the input data
input_scaled = scaler.transform(input_data)

# predict the output
prediction = model.predict(input_scaled)
prediction_probs = prediction[0][0]

if prediction_probs > 0.5:
    st.write(f"The model predicts that the customer will leave the bank with a probability of {prediction_probs:.2f}")  
else:
    st.write(f"The model predicts that the customer will stay with the bank with a probability of {1 - prediction_probs:.2f}")