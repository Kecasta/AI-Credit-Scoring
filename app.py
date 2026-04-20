import streamlit as st
import pandas as pd
import joblib
import os

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="AI Credit Scoring | Fintech Solutions",
    page_icon="💳",
    layout="wide"
)

# --- CUSTOM CSS ---
st.markdown(f"""
    <style>
    .stApp {{
        background-color: #0E1117;
        color: #E0E0E0;
    }}
    .main-header {{
        font-size: 2.5rem;
        font-weight: 800;
        color: #2ECC71;
        margin-bottom: 0rem;
    }}
    .sub-header {{
        font-size: 1.1rem;
        color: #3498DB;
        margin-bottom: 2rem;
    }}
    .metric-card {{
        background-color: #1E2227;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #2ECC71;
        margin-bottom: 1rem;
    }}
    .stButton>button {{
        background-color: #2ECC71;
        color: white;
        width: 100%;
        border-radius: 5px;
        height: 3rem;
        font-weight: bold;
    }}
    .stButton>button:hover {{
        background-color: #27AE60;
        border-color: #27AE60;
    }}
    </style>
    """, unsafe_allow_html=True)

# --- LOADING ASSETS ---
@st.cache_resource
def load_assets():
    if os.path.exists('models/model.pkl') and os.path.exists('models/mappings.joblib'):
        model = joblib.load('models/model.pkl')
        mappings = joblib.load('models/mappings.joblib')
        return model, mappings
    return None, None

model, mappings = load_assets()

# --- HEADER ---
st.markdown('<div class="main-header">Nexus Credit AI</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Advanced Risk Assessment & Predictive Scoring</div>', unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.header("📋 User Profile")
    st.info("Input customer data to calculate risk profile.")
    
    age = st.slider("Age", 18, 70, 35)
    gender = st.selectbox("Gender", ["Male", "Female"])
    income = st.number_input("Annual Income (USD)", min_value=10000, max_value=250000, value=50000, step=1000)
    education = st.selectbox("Education Level", ["High School", "Bachelor", "Master", "Doctorate"])
    marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
    children = st.number_input("Number of Children", 0, 10, 0)
    home_ownership = st.selectbox("Home Ownership", ["Rented", "Owned", "Mortgage"])
    
    st.markdown("---")
    predict_btn = st.button("🚀 Calculate Credit Score")

# --- MAIN CONTENT ---
if model is not None and mappings is not None:
    if predict_btn:
        # Preparation
        input_data = {
            'Age': age,
            'Gender': gender,
            'Income': income,
            'Education': education,
            'Marital Status': marital_status,
            'Number of Children': children,
            'Home Ownership': home_ownership
        }
        
        # Mapping (Vibe Coding Protocol)
        encoded_data = input_data.copy()
        for col, mapping in mappings.items():
            if col in encoded_data:
                encoded_data[col] = mapping.get(encoded_data[col], 0)
        
        # Prediction
        features = pd.DataFrame([encoded_data])
        features = features[['Age', 'Gender', 'Income', 'Education', 'Marital Status', 'Number of Children', 'Home Ownership']]
        
        prediction_code = model.predict(features)[0]
        
        # Reverse mapping for display
        inv_score_mapping = {v: k for k, v in mappings['Credit Score'].items()}
        result = inv_score_mapping.get(prediction_code, "Unknown")
        
        # Display Result
        st.subheader("Assessment Result")
        
        # Color coding
        color = "#2ECC71" # High
        if result == "Low":
            color = "#E74C3C"
        elif result == "Average":
            color = "#3498DB"
            
        st.markdown(f"""
            <div style="background-color: {color}; padding: 30px; border-radius: 15px; text-align: center; color: white;">
                <h1 style="margin:0;">{result.upper()}</h1>
                <p style="margin:0; font-size: 1.2rem;">Credit Score Category</p>
            </div>
            """, unsafe_allow_html=True)
            
        # Insights
        st.markdown("### Profile Analysis")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Income Bracket", f"${income:,.0f}")
        with col2:
            st.metric("Education Level", education)
        with col3:
            st.metric("Dependents", children)
            
        st.success("✅ Prediction generated successfully based on current model weights.")
    else:
        st.warning("👈 Enter candidate details in the sidebar and click 'Calculate Credit Score'")
else:
    st.error("🚨 Model not found! Please run 'python src/train.py' first.")
    if st.button("🔧 Run Training Pipeline"):
        with st.spinner("Training model..."):
            os.system("python src/generate_data.py")
            os.system("python src/train.py")
            st.rerun()

# --- FOOTER ---
st.markdown("---")
st.caption("AI Credit Scoring Portal | Developed for Vibe Coding Environment")
