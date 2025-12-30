import streamlit as st
import pickle
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

from sklearn.preprocessing import LabelEncoder

#------------------------ Load models ----------------------

le_churn = pickle.load(open("le_churn.pkl", "rb"))
dl_model = pickle.load(open("dl_model.pkl", "rb"))
lstm_model = pickle.load(open("lstm_model.pkl", "rb"))
le_sentiment = pickle.load(open("le_sentiment.pkl", "rb"))
tokenizer = pickle.load(open("tokenizer.pkl", "rb"))


st.set_page_config(
    page_title="Customer Churn & Sentiment Analysis",
    page_icon="📊",
    layout="wide")

# ---------Sidebar--------------------------

st.sidebar.title("📌 Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Customer Churn Prediction", "Sentiment Analysis"])


#---------------- Churn Prediction Page---------------------

if page == "Customer Churn Prediction":
    st.title("📉 Customer Churn Prediction")

    with st.form("churn_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            gender = st.selectbox("Gender", ["Male", "Female"])
            SeniorCitizen = st.selectbox("Senior Citizen", [0, 1])
            PhoneService = st.selectbox("Phone Service", ["Yes", "No"])
            MultipleLines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])

        with col2:
            InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
            StreamingTV = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
            StreamingMovies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
            PaperlessBilling = st.selectbox("Paperless Billing", ["Yes", "No"])

        with col3:
            PaymentMethod = st.selectbox("Payment Method",[
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)"])
            
            MonthlyCharges = st.number_input("MonthlyCharges", min_value=0.5, step=0.01)
            
            tenure_binned = st.selectbox("Tenure", ["Low", "Medium", "High"])
            st.write("Low (0-24), Medium (25-48), High (49-72)")
           
            TotalCharges_binned = st.selectbox("Total Charges (Binned)",["Low", "Medium", "High"])
            st.write("Low (0-2895), Medium (2896-5790), High (5791-8700)")

        submitted = st.form_submit_button("Predict Churn")

    if submitted:
        user_data = {
            "gender": gender,
            "SeniorCitizen": SeniorCitizen,
            "PhoneService": PhoneService,
            "MultipleLines": MultipleLines,
            "InternetService": InternetService,
            "StreamingTV": StreamingTV,
            "StreamingMovies": StreamingMovies,
            "PaperlessBilling": PaperlessBilling,
            "PaymentMethod": PaymentMethod,
            "MonthlyCharges": MonthlyCharges,
            "tenure-binned": tenure_binned,
            "TotalCharges-binned": TotalCharges_binned,
            }
        
        df_input = pd.DataFrame([user_data])
        cat_cols = df_input.select_dtypes(include='object').columns

        for col in cat_cols:
            df_input[col] = le_churn.fit_transform(df_input[col])

        churn_probability = dl_model.predict(df_input)[0]

        churn_label = "Yes" if churn_probability > 0.5 else "No"

        st.subheader("🔍 Prediction Result")
        col1, col2 = st.columns(2)

        col1.metric("Churn Prediction", churn_label)


#---------------- Sentiment Analysis Page----------------------

elif page == "Sentiment Analysis":
    st.title("💬 Customer Sentiment Analysis")

    user_text = st.text_area(
        "Enter customer feedback:",
        placeholder="Type customer review or complaint here...")

    if st.button("Analyze Sentiment"):
        if user_text.strip() == "":
            st.warning("Please enter some text.")
        else:
            
            VOCAB_SIZE = 10000
            MAX_LENGTH = 100
            OOV_TOKEN = "<OOV>"

            seq = tokenizer.texts_to_sequences([user_text])
            pad = pad_sequences(seq, maxlen=MAX_LENGTH, padding='post')

            prediction = lstm_model.predict(pad)
            # pred_label_array = le_sentiment.inverse_transform([np.argmax(prediction)])
            # predicted_label = pred_label_array[0]
            predicted_index = np.argmax(prediction)
            predicted_label = le_sentiment.inverse_transform([predicted_index])[0]
           
            st.subheader("📊 Sentiment Result")
            col1, col2 = st.columns(2)

            col1.metric("Sentiment Prediction", predicted_label)
            