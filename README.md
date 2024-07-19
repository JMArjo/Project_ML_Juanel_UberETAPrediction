# Food Delivery Time Prediction

The goal of this project is to build a machine learning model to predict the time taken to deliver the food. Here we deploy the application using Streamlit Community Cloud with an easy to use UI, where the time to deliver is calculated based on some user inputs.

## Table of Contents

- [Dataset](#dataset)
- [Features](#features)
- [Model](#model)
- [Future Ideas](#future-ideas)
- [License](#license)

## Dataset

The dataset used for this project includes the following columns:

- `Delivery_person_Age`: Age of the delivery person
- `Delivery_person_Ratings`: Ratings of the delivery person
- `Weatherconditions`: Weather conditions during delivery
- `Road_traffic_density`: Traffic density on the road
- `Type_of_order`: Type of food order (Snack, Meal, Drinks, Buffet)
- `Type_of_vehicle`: Type of vehicle used for delivery (motorcycle, scooter, electric_scooter)
- `Festival`: Whether the order was placed during a festival
- `City`: Type of city (Urban, Semi-Urban, Metropolitian)
- `Order_Year`, `Order_Month`, `Order_Day`, `Order_DayOfWeek`, `Order_Hour`: Date and time features extracted from the order timestamp
- `multiple_deliveries`: Number of multiple deliveries combined in one trip
- `Time_taken(min)`: Time taken to deliver the order (target variable)

## Features

The model considers the following features for predicting delivery time:

- Distance between restaurant and delivery location
- Historical delivery time
- Extraneous factors: Weather conditions, road traffic density, city, and city type
- Transportation: Vehicle condition, type of vehicle,
- Time-based features: Time of the day, day of the week, festival day  
- Other factors: Delivery partner workload, age, and delivery person’s rating

## Model

The model is built using a Random Forest Regressor. The data is preprocessed using a pipeline that includes scaling numeric features and one-hot encoding discrete features. 

The model is trained and evaluated using the Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and R-squared metrics.

## Future Ideas

- Use more advanced models
- Test with other models and identify which offers better precission
- Refactor to modularize and add some tests to improve QA
- Introduce some hyperparameter tuning.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.