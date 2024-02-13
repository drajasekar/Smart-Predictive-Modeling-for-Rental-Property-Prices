import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
import joblib
import pickle
from sklearn.preprocessing import LabelEncoder
from geopy.distance import geodesic
from math import radians, sin, cos, sqrt, atan2

with open('catboost_modelL.pkl', 'rb') as file:
    catboost_model = pickle.load(file)

# -------------------------------This is the configuration page for our Streamlit Application---------------------------
st.set_page_config(
    page_title="Smart Predictive Modeling for Rental Property Prices",
    page_icon="🏨",
    layout="wide"
)

# -------------------------------This is the sidebar in a Streamlit application, helps in navigation--------------------
with st.sidebar:
    selected = option_menu("Main Menu", ["About Project", "Predictions"],
                           icons=["house", "gear"],
                           styles={"nav-link": {"font": "sans serif", "font-size": "20px", "text-align": "center"},
                                   "nav-link-selected": {"font": "sans serif", "background-color": "#0072b1"},
                                   "icon": {"font-size": "20px"}
                                   }
                           )

# -----------------------------------------------About Project Section--------------------------------------------------
if selected == "About Project":
    st.markdown("# :blue[Smart Predictive Modeling for Rental Property Prices]")
    st.markdown('<div style="height: 50px;"></div>', unsafe_allow_html=True)
    st.markdown("### :blue[Technologies :] Python, Pandas, Numpy, Scikit-Learn, Streamlit, Python scripting, "
                "Machine Learning, Data Preprocessing, Visualization, EDA, Model Building,Model Deployment")
    st.markdown('### :blue[Overview:]')
    st.markdown('The core objective of this initiative is to design and implement a robust predictive modeling system tailored for estimating rental property prices in the prominent cities of Bangalore and Chennai. This project aims to provide invaluable insights to potential renters, landlords, and property investors by analyzing historical rental data and harnessing the power of advanced machine learning algorithms.')

    st.markdown('### Importance:')
    st.markdown('Rental property prices in Bangalore and Chennai are influenced by a multitude of factors, including:')
    st.markdown('- **Location:** Specific neighborhoods or regions within Bangalore and Chennai can significantly impact rental rates, influenced by amenities, infrastructure, and demand.')
    st.markdown('- **Lease Type & Size:** Variations in lease types such as FAMILY,BACHELORand COMPANY.')
    st.markdown('- **Amenities & Facilities:** The presence and quality of amenities like parking spaces, water_supply, gym, and lift.')
    st.markdown('Given the intricate interplay of these influencing elements, the development of a predictive model offers stakeholders a comprehensive, data-driven perspective on rental pricing trends, fostering more informed and strategic decision-making processes.')
    st.markdown("### :blue[Domain :] Real Estate")

# ------------------------------------------------Predictions Section---------------------------------------------------
if selected == "Predictions":
    st.markdown("# :blue[Predicting Results based on Trained Models]")
    st.markdown("### :orange[Predicting Resale Price (Catboost Regressor) (Accuracy: 81%)]")

    

    # Function to perform label encoding
    def encode_data(df):
        ordinal_columns = ['type', 'lease_type', 'facing', 'furnishing', 'water_supply', 'parking', 'building_type']
        label_encoder = LabelEncoder()
        for col in ordinal_columns:
            df[col] = label_encoder.fit_transform(df[col])
        return df

    # Function to calculate Haversine distance
    def haversine_distance(lat1, lon1, lat2, lon2):
        R = 6371  # Radius of the Earth in kilometers
        dlat = radians(lat2 - lat1)
        dlon = radians(lon2 - lon1)
        a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))
        distance = R * c
        return distance

    # Function to calculate distances from central points and create new features
    def calculate_distances(row):
        central_point_1 = {'latitude': 13.0827, 'longitude': 80.2707}
        central_point_2 = {'latitude': 12.9716, 'longitude': 77.5946}
        distance_from_center_1 = haversine_distance(
        central_point_1['latitude'], central_point_1['longitude'],
        radians(row['latitude']), radians(row['longitude'])
        )
        distance_from_center_2 = haversine_distance(
        central_point_2['latitude'], central_point_2['longitude'],
        radians(row['latitude']), radians(row['longitude'])
        )
        return pd.Series([distance_from_center_1, distance_from_center_2], index=['distance_from_center_1', 'distance_from_center_2'])


    def predict_rent(catboost_model, input_df_encoded):
        prediction = catboost_model.predict(input_df_encoded)
        return prediction

    def main():
        # Set the title of the Streamlit app
        st.title('House Rent Prediction')

        # User input
        type = st.selectbox('type', ['BHK2', 'BHK3', 'BHK1', 'RK1', 'BHK4', 'BHK4PLUS', '1BHK1', 'bhk2', 'bhk3'])
        latitude = st.number_input('latitude', min_value=-90.0, max_value=90.0, step=0.0000001, format="%.1f")
        longitude = st.number_input('longitude', min_value=-180.0, max_value=180.0, step=0.0000001, format="%.1f")
        lease_type = st.selectbox('Lease Type', ['ANYONE', 'FAMILY', 'BACHELOR', 'COMPANY'])
        gym = st.selectbox('gym', ['1','0'])
        lift = st.selectbox('lift', ['1','0'])
        swimming_pool = st.selectbox('swimming_pool', ['1','0'])
        negotiable = st.selectbox('negotiable', ['1','0'])
        furnishing = st.selectbox('Furnishing', ['SEMI_FURNISHED', 'FULLY_FURNISHED', 'NOT_FURNISHED'])
        parking = st.selectbox('Parking', ['BOTH', 'TWO_WHEELER', 'NONE', 'FOUR_WHEELER'])
        property_size = st.number_input('property_size')
        property_age = st.number_input('property_age')
        bathroom = st.number_input('Number of Bathrooms', min_value=1, max_value=21, value=1)
        facing = st.selectbox('Facing', ['NE', 'E', 'S', 'N', 'SE', 'W', 'NW', 'SW'])
        cup_board = st.number_input('cup_board')
        floor = st.number_input('floor')
        water_supply = st.selectbox('Water Supply', ['CORPORATION', 'CORP_BORE', 'BOREWELL'])
        building_type = st.selectbox('Building Type', ['AP', 'IH', 'IF', 'GC'])
        balconies = st.number_input('balconies')
        activation_year = st.number_input('Activation Year', min_value=1900, max_value=2023, value=2022)
        

        # Create a dictionary to hold the input data
        input_data = {
            'type': type,
            'latitude': latitude,
            'longitude': longitude,
            'lease_type': lease_type,
            'gym': gym,
            'lift': lift,
            'swimming_pool': swimming_pool,
            'negotiable': negotiable,
            'furnishing': furnishing,
            'parking': parking,
            'property_size': property_size,
            'property_age': property_age,
            'bathroom': bathroom,
            'facing': facing,
            'cup_board': cup_board,
            'floor': floor,
            'water_supply': water_supply,
            'building_type': building_type,
            'balconies': balconies,
            'activation_year': activation_year,
            }

        # Convert input data to a DataFrame
        input_df = pd.DataFrame([input_data])

        # Calculate distances from central points
        input_df[['distance_from_center_1', 'distance_from_center_2']] = input_df.apply(calculate_distances, axis=1)

       # Encode the input data
        input_df_encoded = encode_data(input_df.copy())

        # Make predictions
        if st.button('Predict'):
            prediction = predict_rent(catboost_model, input_df_encoded)
            st.write(f'Predicted Rent: {prediction[0]:.2f} INR')  # Display the predicted rent value

if __name__ == '__main__':
    main()
