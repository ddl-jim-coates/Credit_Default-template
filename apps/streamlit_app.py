import streamlit as st
from streamlit import components
import numpy as np
import pandas as pd
import pickle
import time
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter
import seaborn as sns
import requests
import shap
import xgboost as xgb
import os

# Load the pickled model
with open("/mnt/code/models/xgb_clf.pkl", "rb") as model_file:
    xgc = pickle.load(model_file)

st.set_page_config(layout="wide")

####################
### INTRODUCTION ###
####################

row0_spacer1, row0_1, row0_spacer2, row0_2, row0_spacer3 = st.columns((.1, 2.3, .1, 1.3, .1))
with row0_1:
    st.title('Credit Card Default Dashboard')
with row0_2:
    st.text("")
row3_spacer1, row3_1, row3_spacer2 = st.columns((.1, 3.2, .1))
with row3_1:
    st.markdown("")
    
#################
### SELECTION ###
#################

st.sidebar.text('')
st.sidebar.text('')
st.sidebar.text('')

### INPUT FORM ###
st.sidebar.subheader("**Enter the application inputs to view the default scores.**")
st.sidebar.subheader("")
with st.sidebar.form("my_form"):
    PAY_0 = st.number_input('Repayment status last month', min_value=-1, max_value=9)
    PAY_2 = st.number_input('Repayment status 2 months ago', min_value=-1, max_value=9)
    PAY_3 = st.number_input('Repayment status 3 months ago', min_value=-1, max_value=9)
    PAY_4 = st.number_input('Repayment status 4 months ago', min_value=-1, max_value=9)
    LIMIT_BAL = st.number_input('Credit Limit', min_value=0)
    BILL_AMT1 = st.number_input('Bill Amount', min_value=0)
    scored = st.form_submit_button("Score")

if scored:
    # Define column names for our dataframe
    column_names = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "LIMIT_BAL", "BILL_AMT1"]
        
    # Create DataFrame with the input values
    df = pd.DataFrame([[PAY_0, PAY_2, PAY_3, PAY_4, LIMIT_BAL, BILL_AMT1]], 
                     columns=column_names)
    
    # Convert all columns to integers
    df = df.astype(int)
    
    # Prepare the scoring request
    setup_dict = {
        'PAY_0': int(df.iloc[0]['PAY_0']),
        'PAY_2': int(df.iloc[0]['PAY_2']),
        'PAY_3': int(df.iloc[0]['PAY_3']),
        'PAY_4': int(df.iloc[0]['PAY_4']),
        'LIMIT_BAL': int(df.iloc[0]['LIMIT_BAL']),
        'BILL_AMT1': int(df.iloc[0]['BILL_AMT1'])
    }
    scoring_request = {'data': setup_dict}
    
    try:
        # Make API request
        response = requests.post(os.environ['API_URL'],
            auth=(
                os.environ['API_PASSWORD'],
                os.environ['API_PASSWORD']
            ),
            json=scoring_request
        )
        
        if response.status_code == 200:
            result = response.json().get('result')[0]
            
            # Display prediction result
            st.subheader('Model Prediction:')
            if result == 1:
                st.markdown(":green[REPAYMENT LIKELY]")
            else:
                st.markdown(":red[HIGH RISK OF DEFAULT]")
            
            # Create SHAP explainer
            explainer = shap.TreeExplainer(xgc)
            shap_values = explainer.shap_values(df)
            
            # If shap_values is a list (for XGBClassifier), take the second element for the positive class
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
            
            # Create columns for feature importance and prediction explanation
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader('Feature Importance:')
                feature_importance = pd.DataFrame({
                    'Feature': df.columns,
                    'Weight': np.abs(shap_values).mean(axis=0)
                })
                feature_importance = feature_importance.sort_values('Weight', ascending=False)
                
                st.dataframe(
                    feature_importance.style.background_gradient(
                        axis=0, 
                        gmap=feature_importance['Weight'], 
                        subset=['Feature', 'Weight'], 
                        cmap='Greens'
                    ),
                    hide_index=True
                )
            
            with col2:
                st.subheader('Model Prediction Explanation:')
                prediction_explanation = pd.DataFrame({
                    'Feature': df.columns,
                    'Weight': shap_values[0],
                    'Value': df.iloc[0].values
                })
                prediction_explanation = prediction_explanation.sort_values('Weight', ascending=False)
                
                st.dataframe(
                    prediction_explanation.style.background_gradient(
                        axis=0, 
                        gmap=prediction_explanation['Weight'], 
                        subset=['Feature', 'Weight'], 
                        cmap='RdYlGn'
                    ),
                    hide_index=True
                )
            
            # Add SHAP summary plot
            st.subheader('SHAP Summary Plot:')
            fig, ax = plt.subplots()
            shap.summary_plot(shap_values, df, show=False)
            st.pyplot(fig)
            
        else:
            st.error(f"API call failed with status code: {response.status_code}")
            st.error(f"Response content: {response.content}")
            
    except Exception as e:
        st.error(f"Error making API call: {str(e)}")