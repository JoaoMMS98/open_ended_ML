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
with st.expander('Prediction'):
    Accident_Date = st.date_input("Accident Date", datetime.date(2024, 12, 11))

    Age_at_njury = st.number_input("Age at Injury")

    Alternative_Dispute_Resolution =st.selectbox(" Alternative Dispute Resolution",("Y", "N ", "U"))

    Average_Weekly_Wage = st.st.number_input("Average Weekly Wage")

    Birth_Year = st.st.number_input("Birth Year")

    IME_4_Count = st.st.number_input("IME-4 Count")


    
    

# Make a prediction using the model
    if st.button('Predict'):
        prediction = model.predict([[input_value]])
        st.write(f'Prediction: {prediction[0]}')
    
st.divider()
