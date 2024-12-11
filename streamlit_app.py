import streamlit as st
import pickle as pkl
import pandas

def load_pickle(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)


st.title('To Grant or Not To Grant')

with st.expander:
    try:
        model = load_pickle('model.pkl')
    except Exception as e:
        st.error(f"Error loading pickle file: {e}")
    
    Accident_Date = st.date_input("Accident Date", datetime.date(2024, 12, 11))
    
    

# Make a prediction using the model
    if st.button('Predict'):
        prediction = model.predict([[input_value]])
        st.write(f'Prediction: {prediction[0]}')
    
st.divider()
