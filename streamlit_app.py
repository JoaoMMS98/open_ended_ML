import streamlit as st
import pickle as pkl
import pandas
import datetime

def load_pickle(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)


st.title('To Grant or Not To Grant')

try:
    model = load_pickle('model.pkl')
except Exception as e:
    st.error(f"Error loading pickle file: {e}")

st.header('Input your data here',divider="red")

st.subheader("These subheaders have rotating dividers", divider="red")
Age_at_njury = st.slider("Insert your value", 0, 117, 42)

st.subheader("These subheaders have rotating dividers", divider="red")
Accident_Date = st.date_input("Insert your value", datetime.date(2024, 12, 11))

st.subheader("These subheaders have rotating dividers", divider="red")
Average_Weekly_Wage = st.slider("Insert your value", 0, 2828079, 491.0892)

st.subheader("These subheaders have rotating dividers", divider="red")
Birth_Year = st.slider("Insert your value", 0, 2018, 1977)

st.subheader("IME-4 Count", divider="red")
IME_4_Count = st.slider("Insert your value", 1, 73, 2)


    
    

# Make a prediction using the model
if st.button('Predict'):
    prediction = model.predict([[input_value]])
    st.write(f'Prediction: {prediction[0]}')
    
st.divider()
