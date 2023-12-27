import streamlit as st
from streamlit_option_menu import option_menu
import json
import requests
import pandas as pd
from geopy.distance import geodesic
import statistics
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV


# -------------------------------This is the configuration page for our Streamlit Application---------------------------
st.set_page_config(
    page_title="Smart Predictive Modeling for RentalProperty Prices",
    page_icon="🏨",
    layout="wide"
)

# -------------------------------This is the sidebar in a Streamlit application, helps in navigation--------------------
with st.sidebar:
    selected = option_menu("Main Menu", ["About Project", "Predictions"],
                           icons=["house", "gear"],
                           styles={"nav-link": {"font": "sans serif", "font-size": "20px", "text-align": "centre"},
                                   "nav-link-selected": {"font": "sans serif", "background-color": "#0072b1"},
                                   "icon": {"font-size": "20px"}
                                   }
                           )

# -----------------------------------------------About Project Section--------------------------------------------------
if selected == "About Project":
    st.markdown("# :blue[Smart Predictive Modeling for RentalProperty Prices]")
    st.markdown('<div style="height: 50px;"></div>', unsafe_allow_html=True)
    st.markdown("### :blue[Technologies :] Python, Pandas, Numpy, Scikit-Learn, Streamlit, Python scripting, "
                "Machine Learning, Data Preprocessing, Visualization, EDA, Model Building, Data Wrangling, "
                "Model Deployment")
    st.markdown('### :blue[Overview:]')
    st.markdown('The core objective of this initiative is to design and implement a robust predictive modeling system tailored for estimating rental property prices in the prominent cities of Bangalore and Chennai. This project aims to provide invaluable insights to potential renters, landlords, and property investors by analyzing historical rental data and harnessing the power of advanced machine learning algorithms.')

    st.markdown('### Importance:')
    st.markdown('Rental property prices in Bangalore and Chennai are influenced by a multitude of factors, including:')
    st.markdown('- **Location:** Specific neighborhoods or regions within Bangalore and Chennai can significantly impact rental rates, influenced by amenities, infrastructure, and demand.')
    st.markdown('- **Lease Type & Size:** Variations in lease types such as FAMILY,BACHELORand COMPANY.')
    st.markdown('- **Amenities & Facilities:** The presence and quality of amenities like parking spaces, water_supply,gymand lift,.')
    st.markdown('Given the intricate interplay of these influencing elements, the development of a predictive model offers stakeholders a comprehensive, data-driven perspective on rental pricing trends, fostering more informed and strategic decision-making processes.')
    st.markdown("### :blue[Domain :] Real Estate")

# ------------------------------------------------Predictions Section---------------------------------------------------
if selected == "Predictions":
    st.markdown("# :blue[Predicting Results based on Trained Models]")
    st.markdown("### :orange[Predicting Resale Price (Regression Task) (Accuracy: 79%)]")

    # Import necessary libraries
    import streamlit as st
    import pandas as pd
    import numpy as np
    import joblib
    from sklearn.preprocessing import LabelEncoder

    # Load the trained model
    model = joblib.load('rf_model.pkl')  

    # Function to perform label encoding
    def encode_data(df):
        ordinal_columns = ['type', 'lease_type', 'facing', 'furnishing', 'water_supply', 'parking', 'building_type']
        label_encoder = LabelEncoder()
        for col in ordinal_columns:
            df[col] = label_encoder.fit_transform(df[col])
            return df

    # Function to predict rent
    def predict_rent(model, input_data):
        prediction = model.predict(input_data)
        return prediction

    def main():
        # Set the title of the Streamlit app
        st.title('House Rent Prediction')

        latitude=st.number_input('latitude',min_value=-90.0, max_value=90.0, step=0.0000001, format="%.1f")
        longitude=st.number_input('longitude', min_value=-180.0, max_value=180.0, step=0.0000001, format="%.1f")
        property_size=st.number_input('property_size')
        property_age=st.number_input('property_age')
        floor=st.number_input('floor')
        balconies=st.number_input('balconies')
        type = st.selectbox('type', ['BHK2', 'BHK3', 'BHK1', 'RK1', 'BHK4', 'BHK4PLUS', '1BHK1', 'bhk2', 'bhk3'])
        lease_type = st.selectbox('Lease Type', ['ANYONE', 'FAMILY', 'BACHELOR', 'COMPANY'])
        facing = st.selectbox('Facing', ['NE', 'E', 'S', 'N', 'SE', 'W', 'NW', 'SW'])
        water_supply = st.selectbox('Water Supply', ['CORPORATION', 'CORP_BORE', 'BOREWELL'])
        furnishing = st.selectbox('Furnishing', ['SEMI_FURNISHED', 'FULLY_FURNISHED', 'NOT_FURNISHED'])
        parking = st.selectbox('Parking', ['BOTH', 'TWO_WHEELER', 'NONE', 'FOUR_WHEELER'])
        building_type = st.selectbox('Building Type', ['AP', 'IH', 'IF', 'GC'])
        activation_year = st.number_input('Activation Year', min_value=1900, max_value=2023, value=2022)
        bathroom = st.number_input('Number of Bathrooms', min_value=1, max_value=21, value=1)
        gym = st.selectbox('gym', ['1','0'])
        cup_board=st.number_input('cup_board')
        lift = st.selectbox('lift', ['1','0'])
        swimming_pool = st.selectbox('swimming_pool', ['1','0'])
        negotiable = st.selectbox('negotiable', ['1','0'])

    # Create a dictionary to hold the input data
        input_data = {
        'type': type,
        'latitude':latitude,
        'longitude':longitude,
        'lease_type': lease_type,
        'gym': gym,
        'lift': lift,
        'swimming_pool': swimming_pool,
        'negotiable': negotiable,
        'furnishing': furnishing,
        'parking': parking,
        'property_size':property_size,
        'property_age':property_age,
        'bathroom': bathroom,
        'facing': facing,
        'cup_board':cup_board,
        'floor':floor,
        'water_supply': water_supply,
        'building_type': building_type,
        'balconies':balconies,
        'activation_year': activation_year
        }

        # Convert input data to a DataFrame
        input_df = pd.DataFrame([input_data])

        # Encode the input data
        input_df_encoded = encode_data(input_df.copy())

        # Make predictions
        if st.button('Predict'):
            prediction = predict_rent(model, input_df_encoded)
            st.write(f'Predicted Rent: {prediction[0]:.2f} INR')  # Display the predicted rent value

    if __name__ == '__main__':
        main()