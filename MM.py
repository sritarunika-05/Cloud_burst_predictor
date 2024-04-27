# %%
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
import streamlit as st


# %%
df=pd.read_csv("cloudpredictionsystemproject.csv")

# %%






# %%


# %%
numerical_columns = df.select_dtypes(include=['float64']).columns

df[numerical_columns] = df[numerical_columns].fillna(df[numerical_columns].mean())





# %%
string_columns = df.select_dtypes(include=['object']).columns

# Drop rows with null values in string columns
df.dropna(subset=string_columns, inplace=True)

# %%




# %%
# Find columns with float data type
float_columns = df.select_dtypes(include=['float64']).columns

# Iterate through each float column and replace zero values with the column mean
for col in float_columns:
    if df[col].dtype == 'float64':
        # Calculate column mean excluding zero values
        col_mean = df[col][df[col] != 0].mean()
        # Replace zero values with the column mean
        df[col] = df[col].replace(0, col_mean)

# Now zero values in the float columns are replaced with their respective column means


# %%
string_columns = df.select_dtypes(include=['object']).columns

# Initialize LabelEncoder
label_encoder = LabelEncoder()

# Iterate through each string column and convert its values to integers
for col in string_columns:
    df[col] = label_encoder.fit_transform(df[col].astype(str))

# %%


# %%
df.drop(columns=['Date', 'Location'], axis=1, inplace=True)

# %%


# %%
X= df.drop('CloudBurstTomorrow',axis=1)



# %%
y=df['CloudBurstTomorrow']


# %%
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# %%
# Initialize DecisionTreeRegressor with default hyperparameters
model = DecisionTreeClassifier()


# %%
# Train the model on the training data
model.fit(X_train, y_train)

# %%
model.score(X_test,y_test)


# %%
#model.predict([[13.4,22.9,0.600000,5.468232,7.611178,13,44.0,13,14,20.0,24.0,71.0,22.0,1007.7,1007.1,8.000000,4.50993,16.9,21.8,0]])

# %%
import streamlit as st
import pandas as pd
import pickle

# %%
# Define UI components
st.title("Cloud Burst Prediction")
st.write("Enter the parameters below:")


# %%
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

# %%
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


# %%
# Make predictions and display results
if st.button("Predict"):
    prediction = predict_cloud_burst(minimum_temp, maximum_temp, rainfall, evaporation, sunshine, wind_gust_direction,
                                     wind_gust_speed, wind_direction_9am, wind_direction_3pm, wind_speed_9am,
                                     wind_speed_3pm, humidity_9am, humidity_3pm, pressure_9am, pressure_3pm,
                                     cloud_9am, cloud_3pm, temperature_9am, temperature_3pm, cloud_burst_today)
    st.write("Prediction:", prediction)

# %%



