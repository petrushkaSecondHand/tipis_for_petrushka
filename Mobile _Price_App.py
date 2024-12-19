import streamlit as st
import pandas as pd
import joblib
import os

# Define the current directory path
current_dir = os.path.dirname(os.path.abspath(__file__))

# Load the trained model
model_path = os.path.join(current_dir, 'mobile_price_model.pkl')
model = joblib.load(model_path)

# Load feature information
features_info_path = os.path.join(current_dir, 'features_info.pkl')
features_info = joblib.load(features_info_path)

# Define price range mapping with intervals
price_range_mapping = {
    0: {'label': 'Low', 'interval': '$0 - $500'},
    1: {'label': 'Medium', 'interval': '$501 - $1000'},
    2: {'label': 'High', 'interval': '$1001 - $1500'},
    3: {'label': 'Very High', 'interval': '$1501 - $2000'}
}

# Define descriptive labels for features
feature_labels = {
    'battery_power': 'Total energy a battery can store in one time (mAh)',
    'blue': 'Has Bluetooth',
    'clock_speed': 'Speed at which microprocessor executes instructions (GHz)',
    'dual_sim': 'Has dual SIM support',
    'fc': 'Front Camera megapixels',
    'four_g': 'Has 4G',
    'int_memory': 'Internal Memory in Gigabytes',
    'm_dep': 'Mobile Depth in cm',
    'mobile_wt': 'Weight of mobile phone in grams'
}

# Categorize features
numerical_features = [
    'battery_power', 'clock_speed', 'fc', 'int_memory', 'm_dep', 'mobile_wt'
]
binary_features = ['blue', 'dual_sim', 'four_g']

# Function to get user input
def get_user_input():
    user_data = {}
    for feature in features_info['columns']:
        if feature in numerical_features:
            label = feature_labels[feature]
            value = st.number_input(label, value=0.0)
            user_data[feature] = value
        elif feature in binary_features:
            label = feature_labels[feature]
            value = st.checkbox(label)
            user_data[feature] = 1 if value else 0
    # Create DataFrame with columns in the order of features_info['columns']
    input_df = pd.DataFrame([user_data], columns=features_info['columns'])
    return input_df

# Main app
st.title("Mobile Phone Price Range Predictor")

# Get user input
user_input = get_user_input()

# Display entered values
st.write("Entered Values:")
st.write(user_input)

# Prediction section
if st.button("Predict"):
    # Predict price range
    price_range = model.predict(user_input)[0]
    # Get price range information
    price_info = price_range_mapping.get(price_range, {'label': 'Unknown', 'interval': 'N/A'})
    # Display predictions
    st.subheader(f"Predicted Price Range: {price_info['label']}")
    st.write(f"Price Interval: {price_info['interval']}")