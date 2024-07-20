import pandas as pd
import numpy as np
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

import pickle

from haversine import haversine, Unit

# Load the dataset
df = pd.read_csv('data/train.csv')

# Function to calculate haversine distance
def calculate_distance(row):
    restaurant_coords = (row['Restaurant_latitude'], row['Restaurant_longitude'])
    delivery_coords = (row['Delivery_location_latitude'], row['Delivery_location_longitude'])
    return haversine(restaurant_coords, delivery_coords, unit=Unit.KILOMETERS)

# Calculate haversine distance
df['Distance'] = df.apply(calculate_distance, axis=1)

# Convert 'Delivery_person_Age' to numeric, replacing non-numeric values with NaN
df['Delivery_person_Age'] = pd.to_numeric(df['Delivery_person_Age'], errors='coerce')

# Convert 'Delivery_person_Ratings' to float
df['Delivery_person_Ratings'] = df['Delivery_person_Ratings'].astype(float)

# Convert 'multiple_deliveries' to numeric, replacing non-numeric values with NaN
df['multiple_deliveries'] = pd.to_numeric(df['multiple_deliveries'], errors='coerce')

# Extract numeric value from 'Time_taken(min)'
df['Time_taken'] = df['Time_taken(min)'].str.extract('(\d+)').astype(float)

# Convert 'Order_Date' to datetime
df['Order_Date'] = pd.to_datetime(df['Order_Date'], dayfirst=True)

# Handle missing values using loc to avoid chained assignment
df.loc[df['Delivery_person_Age'].isna(), 'Delivery_person_Age'] = df['Delivery_person_Age'].median()
df.loc[df['Delivery_person_Ratings'].isna(), 'Delivery_person_Ratings'] = df['Delivery_person_Ratings'].mean()
df.loc[df['multiple_deliveries'].isna(), 'multiple_deliveries'] = df['multiple_deliveries'].mode()[0]

# Function to clean and parse time data to avoid NaN
def clean_time(time_str):
    try:
        # Strip any leading/trailing whitespace
        time_str = time_str.strip()
        # Check if the time string is valid
        if time_str != 'NaN':
            # Parse the time string to a datetime object
            return datetime.strptime(time_str, '%H:%M:%S').time()
        else:
            return None  # or you can choose to return a default value
    except ValueError:
        return None  # Handle any other parsing errors

# Apply the cleaning function to the 'time' column
# df['Clean_Time_Orderd'] = df['Time_Orderd'].apply(clean_time)

'''
df['Pickup_Time'] = pd.to_datetime(df['Time_Order_picked'], format = '%H:%M:%S') - pd.to_datetime(df['Clean_Time_Orderd'], format = '%H:%M:%S')

df['Pickup_Time'] = df['Pickup_Time'].dt.total_seconds() / 60
df.drop(df[df['Pickup_Time'] < 0].index, inplace=True)

df = df.dropna(subset=['Pickup_Time'])

pickup_times = df['Pickup_Time'].unique()
# print(f"Unique Pickup_Time values: {pickup_times}")
'''

# Perform one-hot encoding
# df = pd.get_dummies(df, columns=['Pickup_Time'], prefix='Pickup')

# Extract features from the datetime columns
# df['Order_Year'] = df['Order_Date'].dt.year
# df['Order_Month'] = df['Order_Date'].dt.month
# df['Order_Day'] = df['Order_Date'].dt.day
# df['Order_DayOfWeek'] = df['Order_Date'].dt.dayofweek
# df['Order_Hour'] = pd.to_datetime(df['Clean_Time_Orderd'], format='%H:%M:%S').dt.hour

order_types = df['Type_of_order'].unique()
# print(f"Unique Type_of_order values: {order_types}")

# df = pd.get_dummies(df, columns=['Type_of_order'], prefix="Order")

# Split the data into features and target
X = df.drop(['ID', 'Delivery_person_ID', 'Restaurant_latitude', 'Restaurant_longitude', 'Delivery_location_latitude', 
             'Delivery_location_longitude', 'Order_Date', 'Time_taken(min)', 'Time_taken', 'Type_of_order', 'Time_Orderd', 'Time_Order_picked'], axis=1)
# print(X)

y = df['Time_taken']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a preprocessing pipeline
numeric_features = ['Delivery_person_Age', 'Delivery_person_Ratings', 'multiple_deliveries', 'Distance', 'Vehicle_condition']
categorical_features = ['Weatherconditions', 'Road_traffic_density', 'Type_of_vehicle', 'Festival', 'City']

numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Create a dictionary of models
models = {
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'XGBoost': XGBRegressor(n_estimators=100, random_state=42),
    'LightGBM': LGBMRegressor(n_estimators=100, random_state=42),
    'CatBoost': CatBoostRegressor(iterations=100, random_state=42, verbose=False)
}

# Create a dictionary to store the R-squared scores
r2_scores = {}

# Iterate over the models
for model_name, model in models.items():
    # Create the pipeline
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('regressor', model)])
    
    # Train the model
    pipeline.fit(X_train, y_train)
    
    # Make predictions on the test set
    y_pred = pipeline.predict(X_test)
    
    # Calculate R-squared score
    r2 = r2_score(y_test, y_pred)
    r2_scores[model_name] = r2
    
    print(f"{model_name} R-squared: {r2}")

# Find the model with the highest R-squared score
best_model_name = max(r2_scores, key=r2_scores.get)
best_model = models[best_model_name]

print(f"\nBest model: {best_model_name}")

# Create the pipeline with the best model
best_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                ('regressor', best_model)])

# Train the best model
best_pipeline.fit(X_train, y_train)

# Save the trained model
with open('data/model.pkl', 'wb') as file:
    pickle.dump(best_pipeline, file)