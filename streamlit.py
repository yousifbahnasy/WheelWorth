import streamlit as st
import pandas as pd
import joblib

# Load the saved preprocessing pipeline and trained model
preprocessing = joblib.load(r'C:\Users\yousif\project\preprocessing_pipeline.joblib')
rf_model = joblib.load(r'C:\Users\yousif\project\rf_model.joblib')

# Define the function to preprocess and predict
def preprocess_and_predict(input_data):
    # Preprocess the input data
    input_transformed = preprocessing.transform(input_data)
    
    # Make predictions using the trained model
    prediction = rf_model.predict(input_transformed)
    
    return prediction

# Streamlit app layout
st.title('Car Price Prediction App')

# Collect user inputs
make = st.selectbox('Make', ['BMW', 'Audi', 'Mercedes-Benz', 'Toyota', 'Honda'])
model = st.text_input('Model', '1 Series')
year = st.slider('Year', 1990, 2024, 2012)
engine_fuel_type = st.selectbox('Engine Fuel Type', ['Regular', 'Premium', 'Diesel', 'Electric'])
hp = st.number_input('Horsepower (HP)', min_value=50.0, max_value=1000.0, value=320.0)
cylinders = st.number_input('Cylinders', min_value=2.0, max_value=16.0, value=6.0)
transmission = st.selectbox('Transmission', ['MANUAL', 'AUTOMATIC', 'AUTOMATED_MANUAL'])
drive_mode = st.selectbox('Drive Mode', ['rear wheel drive', 'front wheel drive', 'all wheel drive'])
number_of_doors = st.number_input('Number of Doors', min_value=2.0, max_value=5.0, value=2.0)
vehicle_size = st.selectbox('Vehicle Size', ['Compact', 'Midsize', 'Large'])
vehicle_style = st.selectbox('Vehicle Style', ['Coupe', 'Convertible', 'Sedan', 'SUV'])
mpg_h = st.number_input('MPG (Highway)', min_value=10, max_value=100, value=30)
mpg_c = st.number_input('MPG (City)', min_value=10, max_value=100, value=22)
popularity = st.number_input('Popularity', min_value=0, max_value=10000, value=3916)
age_of_car = st.number_input('Age of Car', min_value=0, max_value=30, value=12)

# Button to predict
if st.button('Predict'):
    # Create a DataFrame for the new input
    new_input_data = {
        'Make': [make],
        'Model': [model],
        'Year': [year],
        'Engine Fuel Type': [engine_fuel_type],
        'HP': [hp],
        'Cylinders': [cylinders],
        'Transmission': [transmission],
        'Drive Mode': [drive_mode],
        'Number of Doors': [number_of_doors],
        'Vehicle Size': [vehicle_size],
        'Vehicle Style': [vehicle_style],
        'MPG-H': [mpg_h],
        'MPG-C': [mpg_c],
        'Popularity': [popularity],
        'Age_of_Car': [age_of_car]
    }

    # Convert input data to a DataFrame
    new_input_df = pd.DataFrame(new_input_data)
    
    # Get the predicted price
    predicted_price = preprocess_and_predict(new_input_df)
    
    # Display the prediction
    st.write(f"The predicted price of the car is: ${predicted_price[0]:,.2f}")
