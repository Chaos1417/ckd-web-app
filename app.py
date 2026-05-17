import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- 1. PAGE CONFIG & UI STYLING (Glassmorphism & Animations) ---
st.set_page_config(page_title="CKD Predictor", page_icon="💧", layout="centered")

# Custom CSS for glowy liquid glass and slide-in animations
st.markdown("""
<style>
    /* Animated Gradient Background */
    .stApp {
        background: linear-gradient(-45deg, #0f2027, #203a43, #2c5364, #0b132b);
        background-size: 400% 400%;
        animation: gradientBG 15s ease infinite;
        color: white;
    }
    @keyframes gradientBG {
        0% {background-position: 0% 50%;}
        50% {background-position: 100% 50%;}
        100% {background-position: 0% 50%;}
    }
    
    /* Glassmorphism Container */
    .glass-container {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(15px);
        -webkit-backdrop-filter: blur(15px);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        padding: 30px;
        margin-bottom: 20px;
        animation: slideUp 0.8s ease-out;
    }
    @keyframes slideUp {
        from {opacity: 0; transform: translateY(30px);}
        to {opacity: 1; transform: translateY(0);}
    }

    /* Glowy Button */
    .stButton>button {
        background: linear-gradient(90deg, #00C9FF 0%, #92FE9D 100%);
        border: none;
        color: black;
        font-weight: bold;
        border-radius: 30px;
        padding: 10px 24px;
        transition: all 0.3s ease;
        box-shadow: 0 0 15px rgba(0, 201, 255, 0.5);
    }
    .stButton>button:hover {
        transform: translateY(-2px) scale(1.02);
        box-shadow: 0 0 25px rgba(146, 254, 157, 0.8);
    }
</style>
""", unsafe_allow_html=True)

# --- 2. LOAD MODELS ---
@st.cache_resource # Caches the models so they don't reload on every button click
def load_assets():
    model = joblib.load('ckd_extra_trees_model.pkl')
    scaler = joblib.load('ckd_scaler.pkl')
    encoder = joblib.load('ckd_encoder.pkl')
    return model, scaler, encoder

model, scaler, encoder = load_assets()

# --- 3. UI LAYOUT ---
st.markdown('<div class="glass-container">', unsafe_allow_html=True)
st.title("💧 Chronic Kidney Disease Predictor")
st.write("Enter patient clinical data below. This system utilizes a tuned Extra Trees ensemble model based on the top 12 predictive biomarkers.")
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="glass-container">', unsafe_allow_html=True)
col1, col2 = st.columns(2)

with col1:
    sg = st.selectbox("Specific Gravity (sg)", ['1.005', '1.010', '1.015', '1.020', '1.025'])
    rbc = st.selectbox("Red Blood Cells (rbc)", ["normal", "abnormal"])
    bgr = st.number_input("Blood Glucose Random (bgr)", min_value=0.0, value=120.0)
    bu = st.number_input("Blood Urea (bu)", min_value=0.0, value=30.0)
    sc = st.number_input("Serum Creatinine (sc)", min_value=0.0, value=1.0)
    sod = st.number_input("Sodium (sod)", min_value=0.0, value=140.0)

with col2:
    hemo = st.number_input("Hemoglobin (hemo)", min_value=0.0, value=15.0)
    pcv = st.number_input("Packed Cell Volume (pcv)", min_value=0.0, value=40.0)
    rc = st.number_input("Red Blood Cell Count (rc)", min_value=0.0, value=4.5)
    htn = st.selectbox("Hypertension (htn)", ["yes", "no"])
    dm = st.selectbox("Diabetes Mellitus (dm)", ["yes", "no"])
    pe = st.selectbox("Pedal Edema (pe)", ["yes", "no"])

st.markdown('</div>', unsafe_allow_html=True)

# --- 4. PREDICTION LOGIC ---
if st.button("Analyze Patient Data"):
    
    # Define exact column structures from your notebook
    all_cols = ['age', 'bp', 'sg', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba', 'bgr', 
                'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc', 'htn', 'dm', 
                'cad', 'appet', 'pe', 'ane']
    num_cols = ['age', 'bp', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc']
    cat_cols = ['sg', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane']
    rfe_cols = ['sg', 'rbc', 'bgr', 'bu', 'sc', 'sod', 'hemo', 'pcv', 'rc', 'htn', 'dm', 'pe']

    # Create a dummy dataframe with baseline/average values for ALL 24 columns
    input_data = pd.DataFrame(columns=all_cols)
    input_data.loc[0] = 0.0 # Default for numeric
    input_data.loc[0, cat_cols] = 'missing' # Default for categorical
    
    # Overwrite the dummy data with the actual 12 inputs from the user
    input_data['sg'] = sg
    input_data['rbc'] = rbc
    input_data['bgr'] = bgr
    input_data['bu'] = bu
    input_data['sc'] = sc
    input_data['sod'] = sod
    input_data['hemo'] = hemo
    input_data['pcv'] = pcv
    input_data['rc'] = rc
    input_data['htn'] = htn
    input_data['dm'] = dm
    input_data['pe'] = pe

    # Ensure numerics are floats
    for col in num_cols:
        input_data[col] = pd.to_numeric(input_data[col], errors='coerce')

    # Apply preprocessing exactly as done in training
    input_data[cat_cols] = encoder.transform(input_data[cat_cols])
    input_data[num_cols] = scaler.transform(input_data[num_cols])

    # Slice out ONLY the 12 features RFE selected for the model
    final_features = input_data[rfe_cols]

    # Predict
    prediction = model.predict(final_features)
    probability = model.predict_proba(final_features)[0][1]

    st.markdown('<div class="glass-container">', unsafe_allow_html=True)
    if prediction[0] == 1:
        st.error(f"🚨 **High Risk Predicted** (Confidence: {probability*100:.1f}%)")
        st.write("The clinical indicators strongly suggest the presence of Chronic Kidney Disease.")
    else:
        st.success(f"✅ **Low Risk Predicted** (Confidence: {(1-probability)*100:.1f}%)")
        st.write("The clinical indicators do not suggest Chronic Kidney Disease.")
    st.markdown('</div>', unsafe_allow_html=True)