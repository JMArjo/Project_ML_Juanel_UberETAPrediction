import streamlit as st
import pandas as pd
import pickle
import os
import requests

# Load the trained model
with open('data/model.pkl', 'rb') as file:
    model = pickle.load(file)

# Create the Streamlit app
def main():
    st.title('Food Delivery Time Prediction')
    
    # Input fields
    delivery_person_age = st.number_input('Delivery Person Age', min_value=18, max_value=100, value=25)
    delivery_person_ratings = st.number_input('Delivery Person Ratings', min_value=0.0, max_value=5.0, value=4.5)
    weather_conditions = st.selectbox('Weather Conditions', ['Fog', 'Stormy', 'Sandstorms', 'Windy', 'Cloudy', 'Sunny'])
    road_traffic_density = st.selectbox('Road Traffic Density', ['Low', 'Medium', 'High', 'Jam'])
    type_of_vehicle = st.selectbox('Type of Vehicle', ['motorcycle', 'scooter', 'electric_scooter'])
    vehicle_condition = st.number_input('Vehicle Condition', min_value=0, max_value=2, value=1)
    festival = st.selectbox('Festival', ['No', 'Yes'])
    city = st.selectbox('City', ['Urban', 'Semi-Urban', 'Metropolitian'])
    multiple_deliveries = st.number_input('Multiple Deliveries', min_value=0, max_value=3, value=1)
    distance = st.number_input('Distance', min_value=0, max_value=50, value=10)
    
    # Predict button
    if st.button('Predict'):
        # Create a DataFrame with the input data
        input_data = pd.DataFrame({
            'Delivery_person_Age': [delivery_person_age],
            'Delivery_person_Ratings': [delivery_person_ratings],
            'Weatherconditions': [weather_conditions],
            'Road_traffic_density': [road_traffic_density],
            'Type_of_vehicle': [type_of_vehicle],
            'Vehicle_condition': [vehicle_condition],
            'Festival': [festival],
            'City': [city],
            'multiple_deliveries': [multiple_deliveries],
            'Distance': [distance]
        })
        
        # Make the prediction
        prediction = model.predict(input_data)
        
        # Display the predicted time
        st.success(f'The predicted delivery time is {prediction[0]:.2f} minutes.')

if __name__ == '__main__':
    main()