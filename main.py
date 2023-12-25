import pickle

import joblib
import streamlit as st
import numpy as np
import pandas as pd  # To read the file
import warnings

warnings.filterwarnings('ignore')
from sklearn.preprocessing import LabelEncoder  # Label Encoding process (In the preprocessing part)
from sklearn.preprocessing import FunctionTransformer  # Transformation process (In the preprocessing part)
from sklearn.preprocessing import StandardScaler


def preprocess_input(user_input):
    categorical_cols = ['job', 'marital', 'education', 'default', 'contact', 'housing', 'loan', 'month', 'day_of_week',
                        'poutcome']
    label_encoder = LabelEncoder()
    for col in categorical_cols:
        user_input[col] = label_encoder.fit_transform(user_input[col])

    # Negatif skewness olan sütunlarda log transform yapalım
    log_transformer = FunctionTransformer(np.log1p, validate=True)
    columns_to_log = ['age', 'duration', 'campaign', 'previous']
    negatively_skewed = log_transformer.transform(user_input[columns_to_log])
    user_input['age'] = negatively_skewed[:, 0]
    user_input['duration'] = negatively_skewed[:, 1]
    user_input['campaign'] = negatively_skewed[:, 2]
    user_input['previous'] = negatively_skewed[:, 3]

    # Pozitif skewness olan sütunlarda x^2 transform yapalım
    exp_transformer = FunctionTransformer(lambda x: x ** 2, validate=True)
    columns_to_exp = ['nr.employed']
    positively_skewed = exp_transformer.transform(user_input[columns_to_exp])
    user_input['nr.employed'] = positively_skewed[:, 0]

    # Sayısal sütunları standardize edelim
    numeric_cols = ['age', 'duration', 'campaign', 'previous', 'emp.var.rate', 'cons.price.idx', 'cons.conf.idx',
                    'euribor3m', 'nr.employed']
    user_input[numeric_cols] = StandardScaler().fit_transform(user_input[numeric_cols])

    return user_input
def make_prediction(model, user_input):
    # Tahmin yapmak için modeli ve kullanıcı girdisini kullanın
    model_prediction = model.predict(user_input)

    print('Model Prediction is ' + str(model_prediction))
    if model_prediction == 0:
        return 'No'
    else:
        return 'Yes'


def button_onclick(input_model, user_data):
    age_text_input = user_data['age']
    duration_text_input = user_data['duration']
    pdays_text_input = user_data['pdays']
    previous_text_input = user_data['previous']

    for column in input_data.columns:
        for index, value in input_data[column].items():
            if pd.isna(value) or value == '':
                st.error("Please enter valid values for empty fields")
                st.stop()

    user_data['duration'] = float(pdays_text_input)
    user_data['previous'] = float(previous_text_input)
    user_data['age'] = float(age_text_input)
    user_data['duration'] = float(duration_text_input)

    # Butona basıldığında preprocess ve prediction metotlarını çağır
    processed_data = preprocess_input(user_data)
    result = make_prediction(input_model, processed_data)
    return result


if __name__ == '__main__':
    isfirstrun = True


    age = st.text_input("Age", help='Select your age')
    job = st.selectbox("Job", ["blue-collar", "services", "admin.", "entrepreneur", "self-employed", "technician",
                               "management", "student", "retired", "housemaid", "unemployed"], help='type of job')
    marital_status = st.selectbox("Marital Status", ["married", "single", "divorced", "unknown"], help='marital status ')
    education = st.selectbox("Education",
                             ["basic.9y", "high.school", "university.degree", "professional.course", "basic.6y",
                              "basic.4y", "illiterate", "unknown"], help='education level')
    default = st.selectbox("Default", ["no", "yes", "unknown"], help='has credit in default?	')
    housing = st.selectbox("Housing", ["no", "yes", "unknown"], help='has housing loan?	')
    loan = st.selectbox("Loan", ["no", "yes", "unknown"], help='has personal loan?')
    contact = st.selectbox("Contact", ["cellular", "telephone", "unknown"], help='education level')
    month = st.selectbox("Month", ["may", "jun", "nov", "sep", "jul", "aug", "mar", "oct", "apr", "dec"], help='last '
                                                                                                               'contact month of year')
    day_of_week = st.selectbox("Day of week", ["fri", "wed", "mon", "thu", "tue"], help='last contact day of the week	')
    duration = st.text_input('Duration', help='last contact duration, in seconds (numeric)')
    campaign = st.text_input("Campaign", help='number of contacts performed '
                                              'during this campaign and for'
                                              ' this client (numeric, '
                                              'includes last contact)')
    pdays = st.slider("Pdays", min_value=0.0, max_value=999.0, value=300.0,
                          help='Number of days that passed by after the client was last contacted from a previous campaign')
    previous = st.slider("Previous", min_value=0.0, max_value=6.0, value=2.0, help='Number of contacts performed before this campaign and for this client')
    poutcome = st.selectbox("Poutcome", ["nonexistent", "failure", "success"],
                            help='Outcome of the previous marketing campaign')
    emp_var_rate = st.slider("Employment Variation Rate", min_value=-4.0, max_value=2.0, value=0.0, help='Employment variation rate')
    cons_price_idx = st.slider("Consumer Price Index", min_value=0.0, max_value=100.0, value=30.0,
                               help='Consumer price index')
    cons_conf_idx = st.slider("Consumer Confidence Index", min_value=-70.0, max_value=0.0, value=-20.0,
                              help='Consumer confidence index')
    euribor3m = st.slider("Euribor 3 Month Rate", min_value=0.0, max_value=8.0, value=3.0, help='Euribor 3 month rate')
    nr_employed = st.slider("Number of Employees", min_value=0.0, max_value=6000.0, value=2000.0, help='Number of employees')


    # Create a dataframe with the user input
    input_data = pd.DataFrame({
        'age': [age],
        'job': [job],
        'marital': [marital_status],
        'education': [education],
        'default': [default],
        'housing': [housing],
        'loan': [loan],
        'contact': [contact],
        'month': [month],
        'day_of_week': [day_of_week],
        'duration': [duration],
        'campaign': [campaign],
        'pdays': [pdays],
        'previous': [previous],
        'poutcome': [poutcome],
        'emp.var.rate': [emp_var_rate],
        'cons.price.idx': [cons_price_idx],
        'cons.conf.idx': [cons_conf_idx],
        'euribor3m': [euribor3m],
        'nr.employed': [nr_employed],
    })

    # Load the trained model
    model = joblib.load("model.pkl")

    # Butona basılmadan preprocess ve prediction metodlarını çağırma
    if st.button('Make Prediction'):
        # Set isfirstrun to False after the first run
        isfirstrun = False
        result = button_onclick(model, input_data)  # Capture the result

    # Display the prediction only when the button is clicked and isfirstrun is True
    if not isfirstrun:
        st.subheader("Prediction")
        if result == 0:  # Assuming 0 corresponds to 'No' class in your model
            st.write(f"<span style='font-size:20px; color:red'>{'No'}</span>", unsafe_allow_html=True)
        else:
            st.write(f"<span style='font-size:20px; color:green'>{'Yes'}</span>", unsafe_allow_html=True)