import streamlit as st
import pickle
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
class FrequencyEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.freq_maps = {}

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        for col in X.columns:
            self.freq_maps[col] = X[col].value_counts().to_dict()
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        for col in X.columns:
            X[col] = X[col].map(self.freq_maps[col]).fillna(0)
        return X

with open('m_disaster_prediction_model.pkl','rb') as file:
    model = pickle.load(file)
st.title("Disaster Event Severity Prediction System")


disaster_type = st.text_input("disaster_type")
location = st.text_input("location")
severity_level = st.number_input("severity_level")
affected_population = st.number_input("affected_population")
latitude = st.number_input("latitude")
longitude = st.number_input("longitude")
estimated_economic_loss_usd= st.number_input("estimated_economic_loss_usd")
aid_provided = st.text_input("aid_provided")
infrastructure_damage_index = st.number_input("infrastructure_damage_index")
if st.button("Predict Severity"):

    input_data = pd.DataFrame({
        'disaster_type':[disaster_type],
        'location':[location],
        'severity_level':[severity_level],
        'affected_population':[affected_population],
        'latitude':[latitude],
        'longitude':[longitude],
        'estimated_economic_loss_usd':[estimated_economic_loss_usd],
        'aid_provided':[aid_provided],

        'infrastructure_damage_index':[infrastructure_damage_index]
    })

    prediction = model.predict(input_data)

    st.success(f"Predicted Severity: {prediction[0]}")