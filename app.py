import streamlit as st
import pandas as pd
import pickle

# Load the trained model
with open('titanic_model.pkl', 'rb') as file:
    model = pickle.load(file)
    

# Streamlit app
st.title('Titanic Survival Prediction')

# User input
age = st.slider('Age: ', 0, 92)
class_ = st.selectbox('Class: ', [1, 2, 3])
gender = st.selectbox("Female: 0, Male: 1", [0, 1])
fare = st.slider("Fare (ticket cost in $): ", [0, 512])

# Prepare the input data
input_data = pd.DataFrame({
    'Pclass': [class_],
    'Sex': [gender],
    'Age': [age],
    "Fare": [fare]
})

# Predict Survival
prediction = model.predict(input_data)
prediction_proba = prediction[0]  

# Display the result
#st.write(f'Survival Probability: {prediction_proba:.2f}')

if prediction_proba < 0.5:
    st.write('# You dead.')
    st.image("ded.jpg", use_column_width=True)

else:
    st.write('# You are likely to survive. 💪')
    st.image("surv.jpg", use_column_width=True)
