import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder

# Load the trained model
model = joblib.load('best_model.joblib')

# Load feature importance data
feature_importance = pd.read_csv('feature_importance.csv')

st.title('Xernian Soldier Betrayal Prediction System')

st.write("""
This system predicts the likelihood of a soldier betraying the Xernian clan based on various factors.
Input the soldier's information below for an accurate assessment.
""")

# Create two columns for input fields
col1, col2 = st.columns(2)

# Soldier input features
with col1:
    wealth_level = st.slider('Wealth Level', 0, 10000, 5000)
    promise_of_wealth = st.slider('Promise of Wealth', 0, 10, 5)
    rank_options = ['Private', 'Corporal', 'Sergeant', 'Lieutenant', 'Captain']
    rank = st.selectbox('Rank', rank_options)
    reputation = st.slider('Reputation', 0, 10, 5)
    years_of_service = st.slider('Years of Service', 0, 30, 10)

with col2:
    respect_from_peers = st.slider('Respect from Peers', 0, 10, 5)
    temptation_level = st.slider('Temptation Level', 0, 10, 5)
    influence_of_others = st.slider('Influence of Others', 0, 10, 5)
    mental_resilience = st.slider('Mental Resilience', 0, 10, 5)
    stress_levels = st.slider('Stress Levels', 0, 10, 5)

# Prepare input data
input_data = pd.DataFrame({
    'wealth_level': [wealth_level],
    'promise_of_wealth': [promise_of_wealth],
    'rank': [rank],
    'reputation': [reputation],
    'years_of_service': [years_of_service],
    'respect_from_peers': [respect_from_peers],
    'temptation_level': [temptation_level],
    'influence_of_others': [influence_of_others],
    'mental_resilience': [mental_resilience],
    'stress_levels': [stress_levels]
})

# One-hot encode the 'rank' feature
encoder = OneHotEncoder(sparse=False, drop='first')
rank_encoded = pd.DataFrame(encoder.fit_transform(input_data[['rank']]), columns=encoder.get_feature_names_out(['rank']))
input_data = input_data.drop(columns=['rank'])
input_data = pd.concat([input_data, rank_encoded], axis=1)

# Ensure that the input columns match the model's training data
expected_features = ['wealth_level', 'promise_of_wealth', 'reputation', 'years_of_service', 'respect_from_peers',
                     'temptation_level', 'influence_of_others', 'mental_resilience', 'stress_levels', 'rank_Corporal',
                     'rank_Lieutenant', 'rank_Private', 'rank_Sergeant']

# Ensure all expected columns are present in the input data
for feature in expected_features:
    if feature not in input_data:
        input_data[feature] = 0

# Reorder columns to match the model's expected input
input_data = input_data[expected_features]

# Make prediction
if st.button('Predict Betrayal Likelihood'):
    prediction = model.predict_proba(input_data)[0][1]  # Probability of betrayal

    # Determine risk level
    risk_class = "Low" if prediction < 0.3 else "Moderate" if prediction < 0.7 else "High"
    
    # Display prediction
    st.write(f"Betrayal Likelihood: {prediction:.2%}")
    st.write(f"Risk Level: {risk_class}")
    
    # Feature importance visualization
    st.write('### Top Factors Influencing Betrayal Prediction')
    fig, ax = plt.subplots(figsize=(10, 6))
    feature_importance.sort_values('importance', ascending=True).tail(10).plot.barh(x='feature', y='importance', ax=ax)
    plt.title('Top 10 Most Important Features')
    plt.xlabel('Importance')
    plt.ylabel('Features')
    st.pyplot(fig)

st.write("""
---
**Note:** This prediction system is based on historical data and current factors. 
Use this information ethically and in conjunction with other leadership strategies for best results.
""")
