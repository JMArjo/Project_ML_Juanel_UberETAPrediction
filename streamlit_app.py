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
    type_of_order = st.selectbox('Type of Order', ['Snack', 'Meal', 'Drinks', 'Buffet'])
    type_of_vehicle = st.selectbox('Type of Vehicle', ['motorcycle', 'scooter', 'electric_scooter'])
    festival = st.selectbox('Festival', ['No', 'Yes'])
    city = st.selectbox('City', ['Urban', 'Semi-Urban', 'Metropolitian'])
    order_year = st.number_input('Order Year', min_value=2022, max_value=2024, value=2023)
    order_month = st.number_input('Order Month', min_value=1, max_value=12, value=7)
    order_day = st.number_input('Order Day', min_value=1, max_value=31, value=19)
    order_day_of_week = st.number_input('Order Day of Week', min_value=0, max_value=6, value=4)
    order_hour = st.number_input('Order Hour', min_value=0, max_value=23, value=12)
    multiple_deliveries = st.number_input('Multiple Deliveries', min_value=0, max_value=3, value=1)
    
    # Predict button
    if st.button('Predict'):
        # Create a DataFrame with the input data
        input_data = pd.DataFrame({
            'Delivery_person_Age': [delivery_person_age],
            'Delivery_person_Ratings': [delivery_person_ratings],
            'Weatherconditions': [weather_conditions],
            'Road_traffic_density': [road_traffic_density],
            'Type_of_order': [type_of_order],
            'Type_of_vehicle': [type_of_vehicle],
            'Festival': [festival],
            'City': [city],
            'Order_Year': [order_year],
            'Order_Month': [order_month],
            'Order_Day': [order_day],
            'Order_DayOfWeek': [order_day_of_week],
            'Order_Hour': [order_hour],
            'multiple_deliveries': [multiple_deliveries]
        })
        
        # Make the prediction
        prediction = model.predict(input_data)
        
        # Display the predicted time
        st.success(f'The predicted delivery time is {prediction[0]:.2f} minutes.')

if __name__ == '__main__':
    main()