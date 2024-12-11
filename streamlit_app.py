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

    Alternative_Dispute_Resolution = st.selectbox(" Alternative Dispute Resolution",("Y", "N ", "U"))

    Average_Weekly_Wage = st.number_input("Average Weekly Wage")

    Birth_Year = st.number_input("Birth Year")

    IME_4_Count = st.number_input("IME-4 Count")

    Industry_Code = st.number_input("Industry Code")

    WCIO_Cause_of_Injury_Code = st.number_input("WCIO Cause of Injury Code")

    WCIO_Nature_of_Injury_Code = st.number_input(" WCIO Nature of Injury Code")

    WCIO_Part_Of_Body_Code = st.number_input("WCIO Part Of Body Code")

    Agreement_Reached = st.number_input("Agreement Reached")

    Number_of_Dependents = st.number_input("Number of Dependents")

    Assembly_Date = st.date_input("Assembly Date", datetime.date(2024, 12, 11))

    Attorney_Representative = st.selectbox("Attorney/Representative",("Y", "N"))

    C2_Date = st.date_input("  C-2 Date", datetime.date(2024, 12, 11))

    C3_Date = st.date_input("  C-3 Date", datetime.date(2024, 12, 11))

    Carrier_Name = st.text_input("Carrier Name", "STATE INSURANCE FUND")

    Carrier_Type = st.selectbox("Carrier Type",('1A. PRIVATE', '2A. SIF', '4A. SELF PRIVATE', '3A. SELF PUBLIC',
       'UNKNOWN', '5D. SPECIAL FUND - UNKNOWN',
       '5A. SPECIAL FUND - CONS. COMM. (SECT. 25-A)',
       '5C. SPECIAL FUND - POI CARRIER WCB MENANDS'))
    

    
    
    

# Make a prediction using the model
    if st.button('Predict'):
        prediction = model.predict([[input_value]])
        st.write(f'Prediction: {prediction[0]}')
    
st.divider()
