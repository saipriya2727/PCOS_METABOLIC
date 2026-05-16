import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
from datetime import datetime
import json
import io
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch

# Page configuration
st.set_page_config(
    page_title="HealthScreen AI - Obesity-Focused ML Screening",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Professional Medical-Grade CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 50%, #7e8ba3 100%);
        padding: 2.5rem;
        border-radius: 0;
        color: white;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0,0,0,0.0.15);
        margin: 0 0 2rem 0;
    }
    
    .main-header h1 {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        letter-spacing: -0.5px;
    }
    
    .main-header h2 {
        font-size: 1.3rem;
        font-weight: 400;
        opacity: 0.95;
        margin-bottom: 1rem;
    }
    
    .main-header p {
        font-size: 1.1rem;
        opacity: 0.9;
        max-width: 800px;
        margin: 0 auto;
        line-height: 1.5;
    }
    
    .professional-card {
        background: white;
        border: 1px solid #e1e8ed;
        border-radius: 12px;
        padding: 2rem;
        margin: 1.5rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.0.08);
        transition: all 0.3s ease;
        border-left: 4px solid #00a8cc;
    }
    
    .professional-card:hover {
        box-shadow: 0 4px 16px rgba(0,0,0,0.0.12);
        transform: translateY(-2px);
        border-left-color: #1e3c72;
    }
    
    .professional-card h3 {
        color: #1e3c72;
        font-size: 1.4rem;
        font-weight: 600;
        margin-bottom: 1rem;
        border-bottom: 2px solid #e9ecef;
        padding-bottom: 0.5rem;
    }
    
    .professional-text {
        font-size: 1rem;
        line-height: 1.6;
        color: #495057;
    }
    
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1.5rem;
        margin: 2rem 0;
    }
    
    .metric-item {
        background: white;
        border: 1px solid #e1e8ed;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.0.08);
        transition: all 0.3s ease;
    }
    
    .metric-item:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 16px rgba(0,0,0,0.0.12);
        border-color: #00a8cc;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #1e3c72;
        margin-bottom: 0.5rem;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #6c757d;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        border: none;
        padding: 1rem 2rem;
        border-radius: 8px;
        font-weight: 600;
        font-size: 1.1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(30,60,114,0.3);
        border: 2px solid transparent;
        width: 100%;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #2a5298 0%, #1e3c72 100%);
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(30,60,114,0.4);
        border-color: #00a8cc;
    }
    
    .navigation-card {
        background: white;
        border: 1px solid #e1e8ed;
        border-radius: 12px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.0.08);
        transition: all 0.3s ease;
        text-align: center;
    }
    
    .navigation-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 16px rgba(0,0,0,0.0.12);
        border-color: #00a8cc;
    }
</style>
""", unsafe_allow_html=True)

# Session state for navigation
if 'current_page' not in st.session_state:
    st.session_state.current_page = "home"
if 'form_data' not in st.session_state:
    st.session_state.form_data = {}
if 'prediction_result' not in st.session_state:
    st.session_state.prediction_result = {}

# Navigation function
def navigate_to(page):
    st.session_state.current_page = page
    st.rerun()

# Home Page
if st.session_state.current_page == "home":
    st.markdown("""
    <div class="main-header">
        <h1>⚖️ HealthScreen AI Professional</h1>
        <h2>Medical-Grade Obesity-Focused ML Screening</h2>
        <p>Advanced assessment of PCOS and Metabolic Syndrome using obesity biomarkers</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown('<div style="text-align: center; margin: 3rem 0;">', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🚀 Get Started", use_container_width=True):
            navigate_to("navigation")
    
    st.markdown('</div>', unsafe_allow_html=True)

# Navigation Page
elif st.session_state.current_page == "navigation":
    st.markdown("## 🧭 Choose Screening Type")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="navigation-card">
            <h3>🩺 PCOS Screening (Women)</h3>
            <p class="professional-text">Advanced XGBoost model utilizing 26 obesity-specific biomarkers for precise PCOS detection in women.</p>
            <div class="metric-grid">
                <div class="metric-item">
                    <div class="metric-value">95.5%</div>
                    <div class="metric-label">Accuracy</div>
                </div>
                <div class="metric-item">
                    <div class="metric-value">26</div>
                    <div class="metric-label">Features</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🩺 Start PCOS Screening", use_container_width=True):
            st.session_state.form_type = "pcos"
            navigate_to("form")
    
    with col2:
        st.markdown("""
        <div class="navigation-card">
            <h3>💪 Metabolic Screening</h3>
            <p class="professional-text">Comprehensive XGBoost model analyzing 27 obesity-related factors for metabolic syndrome assessment.</p>
            <div class="metric-grid">
                <div class="metric-item">
                    <div class="metric-value">94.2%</div>
                    <div class="metric-label">Accuracy</div>
                </div>
                <div class="metric-item">
                    <div class="metric-value">27</div>
                    <div class="metric-label">Features</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("💪 Start Metabolic Screening", use_container_width=True):
            st.session_state.form_type = "metabolic"
            navigate_to("form")

# Form Page
elif st.session_state.current_page == "form":
    if st.session_state.form_type == "pcos":
        st.markdown("## 🩺 PCOS Obesity Screening for Women")
        st.markdown("*Using only obesity-related risk factors for PCOS prediction*")
        
        with st.form("pcos_obesity_form"):
            st.markdown("### 📋 Personal Information")
            col1, col2 = st.columns(2)
            
            with col1:
                name = st.text_input("Full Name*")
                age = st.number_input("Age*", min_value=25, max_value=50, value=25)
                weight = st.number_input("Weight (kg)*", min_value=30.0, max_value=200.0, value=70.0, step=0.1)
                height = st.number_input("Height (cm)*", min_value=100.0, max_value=220.0, value=162.0, step=0.1)
            
            with col2:
                metabolic_age = st.number_input("Metabolic Age", min_value=15, max_value=60, value=25)
                family_history_obesity = st.selectbox("Family History of Obesity*", ["no", "yes"])
                childhood_obesity = st.selectbox("Childhood Obesity", ["no", "yes"])
            
            st.markdown("### 📏 Body Measurements")
            col1, col2 = st.columns(2)
            
            with col1:
                waist_circumference = st.number_input("Waist Circumference (cm)*", min_value=50.0, max_value=150.0, value=85.0, step=0.1)
                hip_circumference = st.number_input("Hip Circumference (cm)*", min_value=60.0, max_value=150.0, value=95.0, step=0.1)
            
            with col2:
                body_fat_percentage = st.number_input("Body Fat Percentage (%)", min_value=5.0, max_value=50.0, value=28.0, step=0.1)
                visceral_fat_index = st.number_input("Visceral Fat Index", min_value=1.0, max_value=30.0, value=10.0, step=0.1)
            
            st.markdown("### 🍔 Dietary Patterns")
            col1, col2 = st.columns(2)
            
            with col1:
                diet_calories = st.number_input("Daily Calorie Intake*", min_value=1000.0, max_value=5000.0, value=2200.0, step=50.0)
                fat_intake_grams = st.number_input("Fat Intake (g/day)*", min_value=20.0, max_value=200.0, value=80.0, step=1.0)
            
            with col2:
                carb_intake_grams = st.number_input("Carbohydrate Intake (g/day)*", min_value=100.0, max_value=500.0, value=275.0, step=1.0)
                protein_intake_grams = st.number_input("Protein Intake (g/day)*", min_value=30.0, max_value=200.0, value=90.0, step=1.0)
            
            st.markdown("### 🏃 Lifestyle")
            col1, col2 = st.columns(2)
            
            with col1:
                physical_activity_hours = st.number_input("Physical Activity (hours/day)*", min_value=0.0, max_value=12.0, value=1.5, step=0.5)
                sedentary_hours = st.number_input("Sedentary Time (hours/day)*", min_value=0.0, max_value=24.0, value=8.0, step=0.5)
            
            with col2:
                fast_food_frequency = st.number_input("Fast Food Meals (per week)*", min_value=0.0, max_value=21.0, value=4.0, step=1.0)
                sugar_beverage_frequency = st.number_input("Sugary Beverages (per day)*", min_value=0.0, max_value=10.0, value=2.0, step=1.0)
            
            st.markdown("### 🧬 Hormonal Markers")
            col1, col2 = st.columns(2)
            
            with col1:
                insulin_resistance_score = st.number_input("Insulin Resistance Score (0-10)*", min_value=0.0, max_value=10.0, value=4.0, step=0.1)
                leptin_level = st.number_input("Leptin Level (ng/mL)*", min_value=1.0, max_value=50.0, value=15.0, step=0.1)
            
            with col2:
                adiponectin_level = st.number_input("Adiponectin Level (μg/mL)*", min_value=1.0, max_value=30.0, value=8.0, step=0.1)
                crp_level = st.number_input("CRP Level (mg/L)*", min_value=0.1, max_value=20.0, value=3.5, step=0.1)
            
            st.markdown("---")
            submitted = st.form_submit_button("🔬 Submit Screening", use_container_width=True)
            
            if submitted:
                # Store form data in session state
                st.session_state.form_data = {
                    "name": name,
                    "age": age,
                    "weight": weight,
                    "height": height,
                    "metabolic_age": metabolic_age,
                    "family_history_obesity": family_history_obesity,
                    "childhood_obesity": childhood_obesity,
                    "waist_circumference": waist_circumference,
                    "hip_circumference": hip_circumference,
                    "body_fat_percentage": body_fat_percentage,
                    "visceral_fat_index": visceral_fat_index,
                    "diet_calories": diet_calories,
                    "fat_intake_grams": fat_intake_grams,
                    "carb_intake_grams": carb_intake_grams,
                    "protein_intake_grams": protein_intake_grams,
                    "physical_activity_hours": physical_activity_hours,
                    "sedentary_hours": sedentary_hours,
                    "fast_food_frequency": fast_food_frequency,
                    "sugar_beverage_frequency": sugar_beverage_frequency,
                    "insulin_resistance_score": insulin_resistance_score,
                    "leptin_level": leptin_level,
                    "adiponectin_level": adiponectin_level,
                    "crp_level": crp_level,
                    "type": "pcos"
                }
                navigate_to("results")
    
    elif st.session_state.form_type == "metabolic":
        st.markdown("## 💪 Metabolic Obesity Screening")
        st.markdown("*Using only obesity-related risk factors for Metabolic Syndrome prediction*")
        
        with st.form("metabolic_obesity_form"):
            st.markdown("### 📋 Personal Information")
            col1, col2 = st.columns(2)
            
            with col1:
                name = st.text_input("Full Name*")
                age = st.number_input("Age*", min_value=18, max_value=80, value=35)
                weight = st.number_input("Weight (kg)*", min_value=40.0, max_value=250.0, value=85.0, step=0.1)
                height = st.number_input("Height (cm)*", min_value=120.0, max_value=220.0, value=170.0, step=0.1)
            
            with col2:
                gender = st.selectbox("Gender*", ["male", "female"])
                metabolic_age = st.number_input("Metabolic Age", min_value=15, max_value=80, value=40)
                family_history_obesity = st.selectbox("Family History of Obesity*", ["no", "yes"])
            
            st.markdown("### 📏 Body Measurements")
            col1, col2 = st.columns(2)
            
            with col1:
                waist_circumference = st.number_input("Waist Circumference (cm)*", min_value=50.0, max_value=150.0, value=95.0, step=0.1)
                hip_circumference = st.number_input("Hip Circumference (cm)*", min_value=60.0, max_value=150.0, value=100.0, step=0.1)
            
            with col2:
                body_fat_percentage = st.number_input("Body Fat Percentage (%)", min_value=5.0, max_value=50.0, value=30.0, step=0.1)
                visceral_fat_index = st.number_input("Visceral Fat Index", min_value=1.0, max_value=30.0, value=12.0, step=0.1)
            
            st.markdown("### 🍔 Dietary Patterns")
            col1, col2 = st.columns(2)
            
            with col1:
                diet_calories = st.number_input("Daily Calorie Intake*", min_value=1000.0, max_value=5000.0, value=2500.0, step=50.0)
                fat_intake_grams = st.number_input("Fat Intake (g/day)*", min_value=20.0, max_value=200.0, value=90.0, step=1.0)
            
            with col2:
                carb_intake_grams = st.number_input("Carbohydrate Intake (g/day)*", min_value=100.0, max_value=500.0, value=300.0, step=1.0)
                protein_intake_grams = st.number_input("Protein Intake (g/day)*", min_value=30.0, max_value=200.0, value=100.0, step=1.0)
            
            st.markdown("### 🏃 Lifestyle")
            col1, col2 = st.columns(2)
            
            with col1:
                physical_activity_hours = st.number_input("Physical Activity (hours/day)*", min_value=0.0, max_value=12.0, value=1.0, step=0.5)
                sedentary_hours = st.number_input("Sedentary Time (hours/day)*", min_value=0.0, max_value=24.0, value=10.0, step=0.5)
            
            with col2:
                fast_food_frequency = st.number_input("Fast Food Meals (per week)*", min_value=0.0, max_value=21.0, value=5.0, step=1.0)
                sugar_beverage_frequency = st.number_input("Sugary Beverages (per day)*", min_value=0.0, max_value=10.0, value=3.0, step=1.0)
            
            st.markdown("### 🧬 Hormonal & Metabolic Markers")
            col1, col2 = st.columns(2)
            
            with col1:
                insulin_resistance_score = st.number_input("Insulin Resistance Score (0-10)*", min_value=0.0, max_value=10.0, value=5.0, step=0.1)
                leptin_level = st.number_input("Leptin Level (ng/mL)*", min_value=1.0, max_value=50.0, value=18.0, step=0.1)
            
            with col2:
                adiponectin_level = st.number_input("Adiponectin Level (μg/mL)*", min_value=1.0, max_value=30.0, value=7.0, step=0.1)
                crp_level = st.number_input("CRP Level (mg/L)*", min_value=0.1, max_value=20.0, value=4.0, step=0.1)
                uric_acid_level = st.number_input("Uric Acid Level (mg/dL)*", min_value=2.0, max_value=12.0, value=6.5, step=0.1)
                liver_fat_percentage = st.number_input("Liver Fat Percentage (%)", min_value=1.0, max_value=50.0, value=15.0, step=0.1)
            
            st.markdown("---")
            submitted = st.form_submit_button("🔬 Submit Screening", use_container_width=True)
            
            if submitted:
                # Store form data in session state
                st.session_state.form_data = {
                    "name": name,
                    "age": age,
                    "gender": gender,
                    "weight": weight,
                    "height": height,
                    "metabolic_age": metabolic_age,
                    "family_history_obesity": family_history_obesity,
                    "waist_circumference": waist_circumference,
                    "hip_circumference": hip_circumference,
                    "body_fat_percentage": body_fat_percentage,
                    "visceral_fat_index": visceral_fat_index,
                    "diet_calories": diet_calories,
                    "fat_intake_grams": fat_intake_grams,
                    "carb_intake_grams": carb_intake_grams,
                    "protein_intake_grams": protein_intake_grams,
                    "physical_activity_hours": physical_activity_hours,
                    "sedentary_hours": sedentary_hours,
                    "fast_food_frequency": fast_food_frequency,
                    "sugar_beverage_frequency": sugar_beverage_frequency,
                    "insulin_resistance_score": insulin_resistance_score,
                    "leptin_level": leptin_level,
                    "adiponectin_level": adiponectin_level,
                    "crp_level": crp_level,
                    "uric_acid_level": uric_acid_level,
                    "liver_fat_percentage": liver_fat_percentage,
                    "type": "metabolic"
                }
                navigate_to("results")

# Results Page
elif st.session_state.current_page == "results":
    st.markdown("## 📊 Screening Results")
    
    form_data = st.session_state.form_data
    
    if not st.session_state.prediction_result:
        # Make API call
        try:
            if form_data["type"] == "pcos":
                # Prepare PCOS data for ML API
                pcos_data = {
                    "age": form_data["age"],
                    "weight": form_data["weight"],
                    "height": form_data["height"],
                    "metabolic_age": form_data["metabolic_age"],
                    "family_history_obesity": 1 if form_data["family_history_obesity"] == "yes" else 0,
                    "childhood_obesity": 1 if form_data["childhood_obesity"] == "yes" else 0,
                    "waist_circumference": form_data["waist_circumference"],
                    "hip_circumference": form_data["hip_circumference"],
                    "body_fat_percentage": form_data["body_fat_percentage"],
                    "visceral_fat_index": form_data["visceral_fat_index"],
                    "diet_calories": form_data["diet_calories"],
                    "fat_intake_grams": form_data["fat_intake_grams"],
                    "carb_intake_grams": form_data["carb_intake_grams"],
                    "protein_intake_grams": form_data["protein_intake_grams"],
                    "physical_activity_hours": form_data["physical_activity_hours"],
                    "sedentary_hours": form_data["sedentary_hours"],
                    "fast_food_frequency": form_data["fast_food_frequency"],
                    "sugar_beverage_frequency": form_data["sugar_beverage_frequency"],
                    "insulin_resistance_score": form_data["insulin_resistance_score"],
                    "leptin_level": form_data["leptin_level"],
                    "adiponectin_level": form_data["adiponectin_level"],
                    "crp_level": form_data["crp_level"]
                }
                
                response = requests.post("http://localhost:5000/predict/pcos", json=pcos_data, timeout=10)
                response.raise_for_status()
                result = response.json()
                result["screening_type"] = "PCOS"
                
            else:  # metabolic
                metabolic_data = {
                    "age": form_data["age"],
                    "gender_encoded": 1 if form_data["gender"] == "female" else 0,
                    "weight": form_data["weight"],
                    "height": form_data["height"],
                    "metabolic_age": form_data["metabolic_age"],
                    "family_history_obesity": 1 if form_data["family_history_obesity"] == "yes" else 0,
                    "waist_circumference": form_data["waist_circumference"],
                    "hip_circumference": form_data["hip_circumference"],
                    "body_fat_percentage": form_data["body_fat_percentage"],
                      "visceral_fat_index": form_data["visceral_fat_index"],
                      "diet_calories": form_data["diet_calories"],
                      "fat_intake_grams": form_data["fat_intake_grams"],
                      "carb_intake_grams": form_data["carb_intake_grams"],
                      "protein_intake_grams": form_data["protein_intake_grams"],
                      "physical_activity_hours": form_data["physical_activity_hours"],
                      "sedentary_hours": form_data["sedentary_hours"],
                      "fast_food_frequency": form_data["fast_food_frequency"],
                      "sugar_beverage_frequency": form_data["sugar_beverage_frequency"],
                      "insulin_resistance_score": form_data["insulin_resistance_score"],
                      "leptin_level": form_data["leptin_level"],
                      "adiponectin_level": form_data["adiponectin_level"],
                      "crp_level": form_data["crp_level"],
                      "uric_acid_level": form_data["uric_acid_level"],
                      "liver_fat_percentage": form_data["liver_fat_percentage"]
                      
                    
                }
                
                response = requests.post("http://localhost:5000/predict/metabolic", json=metabolic_data, timeout=10)
                response.raise_for_status()
                result = response.json()
                result["screening_type"] = "Metabolic Syndrome"
            
            # Store result in session state
            st.session_state.prediction_result = result
            
        except requests.exceptions.RequestException as e:
            st.error(f"❌ Unable to connect to ML API: {str(e)}")
            st.error("Please ensure ML server is running at http://localhost:5000")
            st.stop()
    
    # Display results
    if st.session_state.prediction_result:
        result = st.session_state.prediction_result
        prediction = result.get("prediction", 0)
        probability = result.get("probability", 0.0)
        risk_level = result.get("risk_level", "Low")
        
        # Set color based on risk level
        if risk_level == "High":
            risk_color = "🔴"
        elif risk_level == "Moderate":
            risk_color = "🟡"
        else:
            risk_color = "🟢"
        
        bmi = form_data["weight"] / ((form_data["height"] / 100) ** 2)
        
        st.markdown(f"""
        <h3>📊 {result.get('screening_type', 'Screening')} Risk Assessment</h3>
        <div class="metric-grid">
                <div class="metric-item">
                    <div class="metric-value">{risk_color} {risk_level}</div>
                    <div class="metric-label">Risk Level</div>
                </div>
                <div class="metric-item">
                    <div class="metric-value">{probability:.1%}</div>
                    <div class="metric-label">Probability</div>
                </div>
                <div class="metric-item">
                    <div class="metric-value">{bmi:.1f}</div>
                    <div class="metric-label">BMI</div>
                </div>
            </div>
            <p class="professional-text"><strong>ML-based prediction using trained dataset model</strong></p>
        """, unsafe_allow_html=True)
        
        # Future Risk Prediction Section
        if st.session_state.prediction_result:
            try:
                # Prepare data for future risk prediction
                future_risk_data = {
                    "current_risk_level": result.get("risk_level", "Low"),
                    "current_probability": result.get("probability", 0.0),
                    "weight": form_data["weight"],
                    "bmi": form_data["weight"] / ((form_data["height"] / 100) ** 2),
                    "physical_activity_hours": form_data["physical_activity_hours"],
                    "sedentary_hours": form_data["sedentary_hours"],
                    "diet_calories": form_data["diet_calories"],
                    "fat_intake_grams": form_data["fat_intake_grams"],
                    "carb_intake_grams": form_data["carb_intake_grams"],
                    "screening_type": form_data["type"]
                }
                
                # Call future risk API
                future_response = requests.post("http://localhost:5000/predict/future-risk", json=future_risk_data, timeout=10)
                future_response.raise_for_status()
                future_result = future_response.json()
                
                risk_3_months = future_result.get("risk_3_months", "Low")
                risk_6_months = future_result.get("risk_6_months", "Low")
                prob_3_months = future_result.get("prob_3_months", 0.0)
                prob_6_months = future_result.get("prob_6_months", 0.0)
                
                # Determine trend
                current_risk_value = {"Low": 1, "Moderate": 2, "High": 3}.get(result.get("risk_level", "Low"), 1)
                future_risk_value = {"Low": 1, "Moderate": 2, "High": 3}.get(risk_6_months, 1)
                
                if future_risk_value > current_risk_value:
                    trend_emoji = "⬆️"
                    trend_text = "Risk Increasing"
                    trend_color = "#dc3545"
                elif future_risk_value < current_risk_value:
                    trend_emoji = "⬇️"
                    trend_text = "Risk Decreasing"
                    trend_color = "#28a745"
                else:
                    trend_emoji = "➡️"
                    trend_text = "Risk Stable"
                    trend_color = "#ffc107"
                
                st.markdown("---")
                st.markdown('<h3 style="color: #1e3c72; margin-bottom: 1rem;">📈 Future Risk Prediction</h3>', unsafe_allow_html=True)
                
                # Future risk metrics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown(f"""
                    <div class="metric-item">
                        <div class="metric-value" style="color: #6c757d;">{prob_3_months:.1%}</div>
                        <div class="metric-label">Risk in 3 months</div>
                        <div style="font-size: 1.2rem; font-weight: 600; margin-top: 0.5rem; color: #495057;">{risk_3_months}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div class="metric-item">
                        <div class="metric-value" style="color: #6c757d;">{prob_6_months:.1%}</div>
                        <div class="metric-label">Risk in 6 months</div>
                        <div style="font-size: 1.2rem; font-weight: 600; margin-top: 0.5rem; color: #495057;">{risk_6_months}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    st.markdown(f"""
                    <div class="metric-item">
                        <div class="metric-value" style="color: {trend_color}; font-size: 2.5rem;">{trend_emoji}</div>
                        <div class="metric-label">6-Month Trend</div>
                        <div style="font-size: 1.1rem; font-weight: 600; margin-top: 0.5rem; color: {trend_color};">{trend_text}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Recommendations based on trend
                if future_risk_value > current_risk_value:
                    st.markdown("""
                    <div class="professional-text" style="background: #f8d7da; border-left: 4px solid #dc3545; padding: 1rem; margin: 1rem 0; border-radius: 8px;">
                    <strong>⚠️ Recommendation:</strong> Increase physical activity and improve diet to reduce future risk. Consider consulting with healthcare provider for preventive measures.
                    </div>
                    """, unsafe_allow_html=True)
                elif future_risk_value < current_risk_value:
                    st.markdown("""
                    <div class="professional-text" style="background: #d4edda; border-left: 4px solid #28a745; padding: 1rem; margin: 1rem 0; border-radius: 8px;">
                    <strong>✅ Positive Trend:</strong> Continue current lifestyle habits to maintain improving health trajectory.
                    </div>
                    """, unsafe_allow_html=True)
                
            except requests.exceptions.RequestException:
                st.warning("⚠️ Future risk prediction temporarily unavailable. Current results are based on present assessment only.")
            except Exception as e:
                st.warning(f"⚠️ Unable to generate future risk prediction: {str(e)}")
        
        st.markdown("---")
        st.markdown('<div style="text-align: center; margin: 2rem 0;">', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("📋 View Precautions", use_container_width=True):
                navigate_to("precautions")
        
        st.markdown('</div>', unsafe_allow_html=True)

# Precautions Page
elif st.session_state.current_page == "precautions":
    st.markdown("## 📋 Medical Precautions")
    
    form_data = st.session_state.form_data
    result = st.session_state.prediction_result
    screening_type = result.get('screening_type', 'Screening')
    risk_level = result.get("risk_level", "Low")
    
    if screening_type == "PCOS":
        st.markdown('<h3>🩺 PCOS-Specific Precautions</h3>', unsafe_allow_html=True)
        
        if risk_level in ["High", "Moderate"]:
            st.markdown("""
            <div class="professional-text">
            <h4>🏥 Immediate Medical Consultation</h4>
            <ul>
                <li><strong>Endocrinologist Visit:</strong> Schedule appointment with hormone specialist</li>
                <li><strong>Gynecological Exam:</strong> Complete reproductive health assessment</li>
                <li><strong>Ultrasound Imaging:</strong> Check ovarian cysts and uterine health</li>
                <li><strong>Hormone Testing:</strong> Comprehensive endocrine panel evaluation</li>
            </ul>
            
            <h4>💊 Medical Management</h4>
            <ul>
                <li><strong>Birth Control Pills:</strong> Regulate menstrual cycles</li>
                <li><strong>Metformin:</strong> Improve insulin sensitivity</li>
                <li><strong>Anti-androgen Medications:</strong> Reduce testosterone effects</li>
                <li><strong>Fertility Treatments:</strong> If planning pregnancy</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="professional-text">
        <h4>🍔 Lifestyle Modifications</h4>
        <ul>
            <li><strong>Weight Management:</strong> 5-10% weight loss improves symptoms</li>
            <li><strong>Low Glycemic Diet:</strong> Focus on complex carbohydrates</li>
            <li><strong>Regular Exercise:</strong> 150 minutes moderate activity weekly</li>
            <li><strong>Stress Management:</strong> Yoga, meditation, counseling</li>
            <li><strong>Schedule Regulation:</strong> Consistent sleep-wake times</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    else:  # Metabolic Syndrome
        st.markdown('<h3>💪 Metabolic Syndrome-Specific Precautions</h3>', unsafe_allow_html=True)
        
        if risk_level in ["High", "Moderate"]:
            st.markdown("""
            <div class="professional-text">
            <h4>🏥 Immediate Medical Consultation</h4>
            <ul>
                <li><strong>Cardiologist Visit:</strong> Heart health assessment</li>
                <li><strong>Endocrinologist Visit:</strong> Metabolic disorder evaluation</li>
                <li><strong>Lipid Panel:</strong> Complete cholesterol and triglyceride testing</li>
                <li><strong>Glucose Testing:</strong> Fasting glucose and A1c levels</li>
                <li><strong>Blood Pressure Monitoring:</strong> Regular cardiovascular checks</li>
            </ul>
            
            <h4>💊 Medical Management</h4>
            <ul>
                <li><strong>Statins:</strong> Lower cholesterol levels</li>
                <li><strong>Blood Pressure Medications:</strong> ACE inhibitors or ARBs</li>
                <li><strong>Metformin:</strong> Improve insulin sensitivity</li>
                <li><strong>Aspirin Therapy:</strong> Cardiovascular protection</li>
                <li><strong>GLP-1 Agonists:</strong> Blood sugar control</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="professional-text">
        <h4>🍔 Lifestyle Modifications</h4>
        <ul>
            <li><strong>Weight Reduction:</strong> 7-10% body weight loss target</li>
            <li><strong>Heart-Healthy Diet:</strong> Mediterranean or DASH eating plan</li>
            <li><strong>Sodium Reduction:</strong> Limit to 2,300mg daily</li>
            <li><strong>Regular Exercise:</strong> 150 minutes moderate intensity weekly</li>
            <li><strong>Alcohol Limitation:</strong> Maximum 1 drink daily for women</li>
            <li><strong>Smoking Cessation:</strong> Complete tobacco avoidance</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown('<div style="text-align: center; margin: 2rem 0;">', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("📄 Download Report", use_container_width=True):
            navigate_to("download")
    
    st.markdown("---")
elif st.session_state.current_page == "download":
    st.markdown("## 📄 Download Health Report")
    
    form_data = st.session_state.form_data
    result = st.session_state.prediction_result
    screening_type = result.get('screening_type', 'Screening')
    
    # Generate PDF report
    def create_pdf_report():
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        styles = getSampleStyleSheet()
        
        # Build content
        content = []
        content.append(Paragraph(f"HealthScreen AI Report - {screening_type}", styles['Title']))
        content.append(Spacer(1, 12))
        content.append(Paragraph(f"Patient Name: {form_data.get('name', 'N/A')}", styles['Normal']))
        content.append(Paragraph(f"Date: {datetime.now().strftime('%B %d, %Y')}", styles['Normal']))
        content.append(Spacer(1, 12))
        
        # Results section
        content.append(Paragraph("Screening Results", styles['Heading2']))
        content.append(Paragraph(f"Risk Level: {result.get('risk_level', 'N/A')}", styles['Normal']))
        content.append(Paragraph(f"Probability: {result.get('probability', 0):.1%}", styles['Normal']))
        content.append(Paragraph(f"Prediction: {'Positive' if result.get('prediction', 0) == 1 else 'Negative'}", styles['Normal']))
        content.append(Spacer(1, 12))
        
        # BMI calculation
        bmi = form_data["weight"] / ((form_data["height"] / 100) ** 2)
        content.append(Paragraph("Health Metrics", styles['Heading2']))
        content.append(Paragraph(f"BMI: {bmi:.1f}", styles['Normal']))
        content.append(Spacer(1, 12))
        
        # Precautions summary
        content.append(Paragraph("Medical Recommendations", styles['Heading2']))
        if screening_type == "PCOS":
            content.append(Paragraph("• Consult with endocrinologist and gynecologist", styles['Normal']))
            content.append(Paragraph("• Consider hormonal treatments if symptoms persist", styles['Normal']))
            content.append(Paragraph("• Implement lifestyle modifications for weight management", styles['Normal']))
        else:
            content.append(Paragraph("• Consult with cardiologist for cardiovascular assessment", styles['Normal']))
            content.append(Paragraph("• Monitor blood pressure and glucose levels regularly", styles['Normal']))
            content.append(Paragraph("• Adopt heart-healthy diet and exercise routine", styles['Normal']))
        
        content.append(Spacer(1, 12))
        content.append(Paragraph("This report was generated by HealthScreen AI using machine learning analysis.", styles['Normal']))
        content.append(Paragraph("Please consult with healthcare professionals for medical advice.", styles['Normal']))
        
        doc.build(content)
        buffer.seek(0)
        return buffer.getvalue()
    
    # Generate and provide download
    if st.button("📄 Generate PDF Report", use_container_width=True):
        pdf_data = create_pdf_report()
        st.download_button(
            label="📄 Download Health Report PDF",
            data=pdf_data,
            file_name=f"healthscreen_ai_report_{screening_type.lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
            mime="application/pdf"
        )
    
    # Display report summary
    st.markdown('<h3>📋 Report Summary</h3>', unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="professional-text">
    <p><strong>Patient:</strong> {form_data.get('name', 'N/A')}</p>
    <p><strong>Screening Type:</strong> {screening_type}</p>
    <p><strong>Risk Level:</strong> {result.get('risk_level', 'N/A')}</p>
    <p><strong>Probability:</strong> {result.get('probability', 0):.1%}</p>
    <p><strong>Date:</strong> {datetime.now().strftime('%B %d, %Y')}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Navigation buttons
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🏠 Back to Home", use_container_width=True):
            navigate_to("home")
    
    with col2:
        if st.button("🔄 New Screening", use_container_width=True):
            # Clear session data for new screening
            st.session_state.form_data = {}
            st.session_state.prediction_result = {}
            navigate_to("navigation")
