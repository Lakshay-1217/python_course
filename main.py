import streamlit as st
import joblib
import numpy as np

# Load the trained model
model = joblib.load('linear_regression_model.pkl')

# Streamlit app
st.title('Linear Regression Model Interface')
Avg_Session_Length= st.number_input('Avg Session Length')
Time_on_App= st.number_input('Time on App')
Time_on_Website= st.number_input('Time on Website')
Length_of_Membership= st.number_input('Length of Membership')

input_features = np.array([[Avg_Session_Length,Time_on_App,Time_on_Website,Length_of_Membership]])

if st.button('Predict'):
    prediction = model.predict(input_features)
    st.write(f'Predicted Value: {prediction[0]}')
