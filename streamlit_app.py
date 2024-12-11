import streamlit as st
import pickle as pkl
import pandas

def load_pickle(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)


st.title('To Grant or Not To Grant')

# Example: Load a model
try:
    model = load_pickle('model.pkl')
except Exception as e:
    st.error(f"Error loading pickle file: {e}")


input_value = st.number_input('Enter a value:')

# Make a prediction using the model
if st.button('Predict'):
    prediction = model.predict([[input_value]])
    st.write(f'Prediction: {prediction[0]}')
