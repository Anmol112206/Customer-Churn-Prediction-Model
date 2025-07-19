import streamlit as st
import pandas as pd
import pickle
from xgboost import XGBClassifier
from helper import MultiColumnLabelEncoder
import warnings
warnings.filterwarnings("ignore")
import os

base_path = os.path.dirname(__file__)

model_path = os.path.join(base_path, 'churn_pipeline.pkl')
xgb_path = os.path.join(base_path, 'xgb_model.json')

with open(model_path, 'rb') as f:
    model = pickle.load(f)

xgb_model = XGBClassifier()
xgb_model.load_model(xgb_path)

st.set_page_config(page_title="Customer Churn Predictor", layout="wide")


st.markdown("""
    <style>
        .stApp {
            background: linear-gradient(135deg, #0c0c0c 0%, #1a1a1a 25%, #2d2d2d 50%, #1a1a1a 75%, #0c0c0c 100%);
            background-size: 400% 400%;
            animation: gradientShift 15s ease infinite;
        }
        
        @keyframes gradientShift {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }
        
        .main-panel {
            background: linear-gradient(145deg, #1e1e1e, #2a2a2a);
            border: 2px solid #333;
            border-radius: 25px;
            padding: 3rem;
            margin: 2rem auto;
            max-width: 1200px;
            box-shadow: 
                0 20px 40px rgba(0,0,0,0.7),
                inset 0 1px 0 rgba(255,255,255,0.1),
                0 0 60px rgba(64, 224, 255, 0.3);
            animation: floatUp 2s ease-out;
        }
        
        @keyframes floatUp {
            from { opacity: 0; transform: translateY(30px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .title {
            font-size: 48px;
            font-weight: 900;
            text-align: center;
            margin-bottom: 30px;
            background: linear-gradient(45deg, #40E0D0, #FF6B6B, #4ECDC4, #45B7D1);
            background-size: 400% 400%;
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            animation: gradientText 6s ease infinite;
            text-shadow: 0 0 30px rgba(64, 224, 255, 0.5);
        }
        
        @keyframes gradientText {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }
        
        .subheader {
            font-size: 28px;
            font-weight: 700;
            color: #E0E0E0;
            margin: 30px 0 20px 0;
            text-align: center;
            text-shadow: 0 2px 4px rgba(0,0,0,0.8);
        }
        
        .stSelectbox > div > div > div,
        .stSlider > div > div > div,
        .stNumberInput > div > div > input {
            background: linear-gradient(145deg, #2a2a2a, #1e1e1e) !important;
            border: 2px solid #444 !important;
            border-radius: 12px !important;
            color: #E0E0E0 !important;
            font-weight: 600 !important;
            box-shadow: inset 0 2px 4px rgba(0,0,0,0.3) !important;
            transition: all 0.3s ease !important;
        }
        
        .stSelectbox > div > div > div:hover,
        .stNumberInput > div > div > input:hover {
            border-color: #40E0D0 !important;
            box-shadow: 0 0 15px rgba(64, 224, 255, 0.4) !important;
        }
        
        .stButton > button {
            background: transparent !important;
            background-size: 200% 200% !important;
            border: none !important;
            border-radius: 25px !important;
            color: white !important;
            font-size: 26px !important;
            font-weight: 800 !important;
            padding: 20px 60px !important;
            box-shadow: none !important;
            animation: buttonPulse 3s ease infinite !important;
            transition: all 0.3s ease !important;
            width: 100% !important;
        }
            
        .stButton > button:hover {
            transform: translateY(-3px) scale(1.05) !important;
            box-shadow: 
                0 12px 35px rgba(255, 107, 107, 0.6),
                0 0 35px rgba(64, 224, 255, 0.5) !important;
        }
        
        @keyframes buttonPulse {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }
        
        .result-box {
            background: linear-gradient(145deg, #2a2a2a, #1e1e1e);
            border: 2px solid #444;
            border-radius: 20px;
            padding: 2rem;
            margin: 2rem auto;
            text-align: center;
            font-size: 24px;
            font-weight: 700;
            color: #E0E0E0;
            box-shadow: 
                0 15px 35px rgba(0,0,0,0.5),
                inset 0 1px 0 rgba(255,255,255,0.1),
                0 0 40px rgba(64, 224, 255, 0.2);
            animation: resultGlow 2s ease infinite;
        }
        
        @keyframes resultGlow {
            0% { box-shadow: 0 15px 35px rgba(0,0,0,0.5), inset 0 1px 0 rgba(255,255,255,0.1), 0 0 40px rgba(64, 224, 255, 0.2); }
            50% { box-shadow: 0 15px 35px rgba(0,0,0,0.5), inset 0 1px 0 rgba(255,255,255,0.1), 0 0 60px rgba(64, 224, 255, 0.4); }
            100% { box-shadow: 0 15px 35px rgba(0,0,0,0.5), inset 0 1px 0 rgba(255,255,255,0.1), 0 0 40px rgba(64, 224, 255, 0.2); }
        }
        
        .stSelectbox > label,
        .stSlider > label,
        .stNumberInput > label {
            color: #E0E0E0 !important;
            font-weight: 600 !important;
            font-size: 16px !important;
            text-shadow: 0 1px 2px rgba(0,0,0,0.8) !important;
        }
        
        .stColumn {
            background: rgba(255,255,255,0.02);
            border-radius: 15px;
            padding: 20px;
            margin: 10px;
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.05);
        }
        
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

with st.form(key="churn_form"):
    st.markdown('<div class="title">Customer Churn Predictor</div>', unsafe_allow_html=True)
    st.markdown('<div class="subheader">✨ Customer Information</div>', unsafe_allow_html=True)

    col1, col2,col3 = st.columns(3)

    with col1:
        gender = st.selectbox("👤 Gender", ["Male", "Female"])
        geography = st.selectbox("🌍 Geography", ["France", "Spain", "Germany"])
        is_active_display = st.selectbox("🔥 Is Active Member", ["Yes", "No"])
        has_card_display = st.selectbox("💳 Has Credit Card", ["Yes", "No"])

    with col2:
        tenure = st.slider("⏳ Tenure (years)", 0, 10, 3)
        num_products = st.slider("📦 Number of Products", 1, 4, 2)
        age = st.slider("📅 Age", 18, 100, 35)

    with col3:
        credit_score = st.number_input("📊 Credit Score", min_value=300, max_value=1000, value=600)
        balance = st.number_input("💰 Balance", min_value=0.0, value=50000.0)
        salary = st.number_input("💵 Estimated Salary", min_value=0.0, value=70000.0)

    is_active = 1 if is_active_display == "Yes" else 0
    has_card = 1 if has_card_display == "Yes" else 0

    col = st.columns(1)[0]
    with col:
        submit = st.form_submit_button("✨ Predict Churn Risk ✨")

    if submit:
        input_df = pd.DataFrame([{
            'Gender': gender,
            'Geography': geography,
            'Age': age,
            'CreditScore': credit_score,
            'Balance': balance,
            'EstimatedSalary': salary,
            'Tenure': tenure,
            'NumOfProducts': num_products,
            'IsActiveMember': is_active,
            'HasCrCard': has_card
        }])

        expected_order = ['CreditScore', 'Geography', 'Gender', 'Age', 'Tenure',
                          'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']
        input_df = input_df[expected_order]

        encoded_input = model.transform(input_df)
        prob = xgb_model.predict_proba(encoded_input)[0][1]
        prediction = int(prob > 0.21)

        if prediction == 1:
            risk_emoji = "🚨"
            risk_message = "Customer has high chances to churn"
            message_color = "#FF6B6B"
        else:
            risk_emoji = "✅"
            risk_message = "Customer has low chances to churn"
            message_color = "#4ECDC4"
        
        result_text = f"""
        <div style="text-align: center;">
            <div style="font-size: 64px; margin-bottom: 20px;">{risk_emoji}</div>
            <div style="font-size: 32px; color: {message_color}; font-weight: 900; text-shadow: 0 0 10px {message_color};">
                {risk_message}
            </div>
        </div>
        """

        st.markdown('<div class="result-box">' + result_text + '</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)