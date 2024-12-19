import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder,OneHotEncoder
import pandas as pd
import pickle

# Load the traiend model
model = tf.keras.models.load_model('model.h5')

# Load the categoriacal encoders and data scaler from pickle file
with open('lable_encode_gender.pkl','rb') as file:
    lable_encode_gender = pickle.load(file)
with open('onehot_encoder_geo.pkl','rb') as file:
    lable_encode_geo = pickle.load(file)
with open('scaler.pkl','rb') as file:
    scaler = pickle.load(file)

# Streamlit app

st.title('Customer churn Prediction')

credit_score = st.number_input('Credit Sore')
geography = st.selectbox('Geography', lable_encode_geo.categories_[0])
gender = st.selectbox('Gender', lable_encode_gender.classes_)
age = st.number_input('Age', 18,92)
tenure = st.slider('Tenure', 0,10) 
balance = st.number_input('Balance') 
num_of_products = st.number_input('NumOfProducts',1,4) 
has_cr_card = st.selectbox('Has Credict card', [0,1])
is_active_member = st.selectbox('Is Active Member', [0,1]) 
estimated_salary= st.number_input('Estimated Salary')

input_data = pd.DataFrame( {
    'CreditScore':[credit_score],
    'Gender': [lable_encode_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure':[tenure], 
    'Balance':[balance], 
    'NumOfProducts':[num_of_products], 
    'HasCrCard':[has_cr_card],
    'IsActiveMember':[is_active_member], 
    'EstimatedSalary':[estimated_salary]
}
)

# one-hot-encoding of Geography 
geo_encoded = lable_encode_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=lable_encode_geo.get_feature_names_out(['Geography']))


# Combine one-hoot encoded columns with input data 
input_df = pd.concat([input_data.reset_index(drop=True), geo_encoded_df],axis=1)


# Scale the input data
input_scaled_data = scaler.transform(input_df)

# Predict churn 
prediction = model.predict(input_scaled_data)
prediction_probability = prediction[0][0]


if prediction_probability > 0.5:
    st.write('The customer is likely to churn.')
else:
    st.write('The customer is not likely to churn.')