import streamlit as st
import joblib
import numpy as np

model = joblib.load('RFclass.pkl')

def main():
    st.title('Machine learning model deployment')

    CreditScore = st.slider('CreditScore', min_value=350.0, max_value = 850.0, value=1.0)
    
    Geography = st.radio('choose 1 (Geography)', ['Germany', 'Spain', 'France'])

    Gender = st.radio('What is your gender?', ['Male', 'Female'])

    Age = st.slider('Age', min_value=18.0, max_value=92.0, value=1.0)

    Balance = st.slider('Balance', min_value=0.0, max_value=250899.0, value=0.1)

    Tenure = st.slider('Tenure', min_value=0.0, max_value=10.0, value=1.0)

    NumOfProducts = st.selectbox('Choose 1', [0, 1, 2, 3, 4])

    HasCrCard = st.radio('Has a Credit card?', ['Yes', 'No'])

    IsActiveMember = st.radio('Is active member?', ['Yes', 'No'])

    EstimatedSalary = st.slider('Estimated Salary', min_value=11.0, max_value=199992.5, value=0.1)

    if st.button('Make Prediction'):
        features = [CreditScore, Geography, Gender, Age, Balance, Tenure, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary]
        result = make_prediction(features)
        st.success(f'The prediction is: {result}')

def make_prediction(features):
    input_array = np.array(features).reshape(1, -1)
    prediction = model.predict(input_array)
    return prediction[0]

if __name__ == '__main__':
    main()
