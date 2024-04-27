import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import OneHotEncoder
from twilio.rest import Client
import keys
import subprocess
# Load the trained model from the pickle file
with open('trained_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Define UI components
st.title("Cloud Burst Prediction")
st.write("Enter the parameters below:")

# Add input fields for parameters
minimum_temp = st.number_input("Minimum Temperature", value=0.0)
maximum_temp = st.number_input("Maximum Temperature", value=0.0)
rainfall = st.number_input("Rainfall", value=0.0)
evaporation = st.number_input("Evaporation", value=0.0)
sunshine = st.number_input("Sunshine", value=0.0)
wind_gust_direction = st.number_input("Wind Gust Direction", value=0.0)
wind_gust_speed = st.number_input("Wind Gust Speed", value=0.0)
wind_direction_9am = st.number_input("Wind Direction 9am", value=0.0)
wind_direction_3pm = st.number_input("Wind Direction 3pm", value=0.0)
wind_speed_9am = st.number_input("Wind Speed 9am", value=0.0)
wind_speed_3pm = st.number_input("Wind Speed 3pm", value=0.0)
humidity_9am = st.number_input("Humidity 9am", value=0.0)
humidity_3pm = st.number_input("Humidity 3pm", value=0.0)
pressure_9am = st.number_input("Pressure 9am", value=0.0)
pressure_3pm = st.number_input("Pressure 3pm", value=0.0)
cloud_9am = st.number_input("Cloud 9am", value=0.0)
cloud_3pm = st.number_input("Cloud 3pm", value=0.0)
temperature_9am = st.number_input("Temperature 9am", value=0.0)
temperature_3pm = st.number_input("Temperature 3pm", value=0.0)
cloud_burst_today = st.number_input("Cloud Burst Today", value=0.0)

# Define prediction function
def predict_cloud_burst(minimum_temp, maximum_temp, rainfall, evaporation, sunshine, wind_gust_direction, wind_gust_speed,
                        wind_direction_9am, wind_direction_3pm, wind_speed_9am, wind_speed_3pm, humidity_9am,
                        humidity_3pm, pressure_9am, pressure_3pm, cloud_9am, cloud_3pm, temperature_9am,
                        temperature_3pm, cloud_burst_today):
    # Format user inputs into a DataFrame
    data = [[
    minimum_temp,
    maximum_temp,
    rainfall,
    evaporation,
    sunshine,
    wind_gust_direction,
    wind_gust_speed,
    wind_direction_9am,
    wind_direction_3pm,
    wind_speed_9am,
    wind_speed_3pm,
    humidity_9am,
    humidity_3pm,
    pressure_9am,
    pressure_3pm,
    cloud_9am,
    cloud_3pm,
    temperature_9am,
    temperature_3pm,
    cloud_burst_today
]
    ]

    # Make predictions using the model
    prediction = model.predict(data)
    return prediction

# Make predictions and display results
if st.button("Predict"):
    prediction = predict_cloud_burst(minimum_temp, maximum_temp, rainfall, evaporation, sunshine, wind_gust_direction,
                                     wind_gust_speed, wind_direction_9am, wind_direction_3pm, wind_speed_9am,
                                     wind_speed_3pm, humidity_9am, humidity_3pm, pressure_9am, pressure_3pm,
                                     cloud_9am, cloud_3pm, temperature_9am, temperature_3pm, cloud_burst_today)
    if prediction==0:
        st.write("Prediction:","THERE IS NO CLOUDBURST TOMORROW")
        python_file_path = "C:\\Users\\ROSHAN\\Downloads\\SIH - Cloud burst prediction\\msg.py"
        subprocess.run(["python", python_file_path])

       
    else:
        st.write("Prediction:","THERE IS  CLOUDBURST TOMORROW")
        python_file_path = "C:\\Users\\ROSHAN\\Downloads\\SIH - Cloud burst prediction\\msg.py"
        subprocess.run(["python", python_file_path])
    # Path to the Python file you want to run
        



    
        
        
            

        