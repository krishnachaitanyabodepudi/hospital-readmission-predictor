"""
Hospital Readmission Prediction - Streamlit Dashboard

Clean, professional dashboard for hospital readmission risk prediction.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Hospital Readmission Predictor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .risk-high { color: #d62728; font-weight: bold; }
    .risk-medium { color: #ff7f0e; font-weight: bold; }
    .risk-low { color: #2ca02c; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

class HospitalReadmissionDashboard:
    def __init__(self):
        self.model = None
        self.feature_names = None
        self.scaler = None
        self.load_models()
    
    def load_models(self):
        try:
            if os.path.exists('models/best_model.joblib'):
                self.model = joblib.load('models/best_model.joblib')
                st.success("✅ Model loaded successfully")
            else:
                st.error("❌ Model file not found. Please train the model first.")
                st.info("💡 Run: python train_and_deploy.py")
            
            if os.path.exists('models/scaler.joblib'):
                self.scaler = joblib.load('models/scaler.joblib')
            else:
                st.warning("⚠️ Scaler not found")
            
            if os.path.exists('data/feature_names.csv'):
                self.feature_names = pd.read_csv('data/feature_names.csv')['feature'].tolist()
            else:
                st.warning("⚠️ Feature names file not found")
        except Exception as e:
            st.error(f"Error loading models: {e}")
            st.info("Please ensure model files are present in the repository.")
    
    def create_header(self):
        st.markdown('<h1 class="main-header">🏥 Hospital Readmission Risk Predictor</h1>', unsafe_allow_html=True)
        st.markdown('<p style="text-align: center; font-size: 1.1rem; color: #666;">AI-powered prediction for patient readmission risk assessment</p>', unsafe_allow_html=True)
    
    def create_sidebar(self):
        st.sidebar.header("📊 Patient Information")
        
        st.sidebar.subheader("Demographics")
        age = st.sidebar.slider("Age", 0, 100, 65)
        gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
        race = st.sidebar.selectbox("Race", ["Caucasian", "AfricanAmerican", "Hispanic", "Asian", "Other"])
        
        st.sidebar.subheader("Medical Information")
        time_in_hospital = st.sidebar.slider("Days in Hospital", 1, 14, 5)
        num_lab_procedures = st.sidebar.slider("Number of Lab Procedures", 0, 100, 43)
        num_procedures = st.sidebar.slider("Number of Procedures", 0, 10, 1)
        num_medications = st.sidebar.slider("Number of Medications", 0, 30, 16)
        number_diagnoses = st.sidebar.slider("Number of Diagnoses", 1, 20, 7)
        
        st.sidebar.subheader("Service Utilization")
        number_outpatient = st.sidebar.slider("Outpatient Visits", 0, 50, 0)
        number_emergency = st.sidebar.slider("Emergency Visits", 0, 50, 0)
        number_inpatient = st.sidebar.slider("Previous Inpatient Stays", 0, 20, 0)
        
        st.sidebar.subheader("Admission Details")
        admission_type = st.sidebar.selectbox("Admission Type", ["Emergency", "Urgent", "Elective"])
        discharge_disposition = st.sidebar.selectbox("Discharge Disposition", ["Home", "Other", "Transfer"])
        admission_source = st.sidebar.selectbox("Admission Source", ["Emergency", "Transfer", "Referral"])
        
        st.sidebar.subheader("Diabetes Management")
        diabetes_med = st.sidebar.selectbox("Diabetes Medication", ["Yes", "No"])
        medication_change = st.sidebar.selectbox("Medication Change", ["Yes", "No"])
        
        st.sidebar.subheader("Lab Results")
        max_glu_serum = st.sidebar.selectbox("Max Glucose Serum", ["Normal", "High", "Not Tested"])
        a1c_result = st.sidebar.selectbox("A1C Result", ["Normal", "High", "Not Tested"])
        
        return {
            'age': age, 'gender': gender, 'race': race,
            'time_in_hospital': time_in_hospital, 'num_lab_procedures': num_lab_procedures,
            'num_procedures': num_procedures, 'num_medications': num_medications,
            'number_diagnoses': number_diagnoses, 'number_outpatient': number_outpatient,
            'number_emergency': number_emergency, 'number_inpatient': number_inpatient,
            'admission_type': admission_type, 'discharge_disposition': discharge_disposition,
            'admission_source': admission_source, 'diabetes_med': diabetes_med,
            'medication_change': medication_change, 'max_glu_serum': max_glu_serum,
            'a1c_result': a1c_result
        }
    
    def preprocess_input(self, input_data):
        """Preprocess input to create all 96 features"""
        if self.feature_names is None:
            self.feature_names = pd.read_csv('data/feature_names.csv')['feature'].tolist()
        
        feature_dict = {}
        
        # Basic features
        feature_dict['time_in_hospital'] = input_data['time_in_hospital']
        feature_dict['num_lab_procedures'] = input_data['num_lab_procedures']
        feature_dict['num_procedures'] = input_data['num_procedures']
        feature_dict['num_medications'] = input_data['num_medications']
        feature_dict['number_outpatient'] = input_data['number_outpatient']
        feature_dict['number_emergency'] = input_data['number_emergency']
        feature_dict['number_inpatient'] = input_data['number_inpatient']
        feature_dict['number_diagnoses'] = input_data['number_diagnoses']
        
        feature_dict['age_numeric'] = input_data['age']
        feature_dict['gender_binary'] = 1 if input_data['gender'] == 'Male' else 0
        feature_dict['diabetes_med_binary'] = 1 if input_data['diabetes_med'] == 'Yes' else 0
        feature_dict['change_binary'] = 1 if input_data['medication_change'] == 'Yes' else 0
        
        # Race encoding
        race_map = {'Asian': 'race_Asian', 'Caucasian': 'race_Caucasian', 
                   'Hispanic': 'race_Hispanic', 'Other': 'race_Other', 
                   'AfricanAmerican': 'race_AfricanAmerican'}
        for race_key, race_col in race_map.items():
            feature_dict[race_col] = 1 if input_data['race'] == race_key else 0
        
        # Admission encoding
        admission_type_map = {'Emergency': 'admission_type_id_2', 'Urgent': 'admission_type_id_3', 
                             'Elective': 'admission_type_id_5'}
        for adm_key, adm_col in admission_type_map.items():
            feature_dict[adm_col] = 1 if input_data['admission_type'] == adm_key else 0
        
        discharge_map = {'Home': 'discharge_disposition_id_2', 'Other': 'discharge_disposition_id_3', 
                        'Transfer': 'discharge_disposition_id_18'}
        for disc_key, disc_col in discharge_map.items():
            feature_dict[disc_col] = 1 if input_data['discharge_disposition'] == disc_key else 0
        
        admission_source_map = {'Emergency': 'admission_source_id_2', 'Transfer': 'admission_source_id_9', 
                               'Referral': 'admission_source_id_4'}
        for src_key, src_col in admission_source_map.items():
            feature_dict[src_col] = 1 if input_data['admission_source'] == src_key else 0
        
        max_glu_map = {'High': 'max_glu_serum_>300', 'Normal': 'max_glu_serum_Norm'}
        for glu_key, glu_col in max_glu_map.items():
            feature_dict[glu_col] = 1 if input_data['max_glu_serum'] == glu_key else 0
        
        a1c_map = {'High': 'A1Cresult_>8', 'Normal': 'A1Cresult_Norm'}
        for a1c_key, a1c_col in a1c_map.items():
            feature_dict[a1c_col] = 1 if input_data['a1c_result'] == a1c_key else 0
        
        # Medication features - distribute medications
        medications = ['metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride', 
                       'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide', 'pioglitazone', 
                       'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone', 'tolazamide', 
                       'examide', 'citoglipton', 'insulin', 'glyburide-metformin', 'glipizide-metformin', 
                       'glimepiride-pioglitazone', 'metformin-rosiglitazone', 'metformin-pioglitazone']
        
        # Medication features - set first N medications to 1 based on total count
        total_meds = input_data['num_medications']
        for i, med in enumerate(medications):
            feature_dict[med] = 1 if i < total_meds else 0
        
        # Calculate derived features that vary with inputs
        feature_dict['patient_service_utilization'] = (input_data['number_outpatient'] + 
                                                       input_data['number_emergency'] + 
                                                       input_data['number_inpatient'])
        feature_dict['medication_changes'] = 1 if input_data['medication_change'] == 'Yes' else 0
        feature_dict['total_medications'] = input_data['num_medications']
        
        # Make values more varied
        feature_dict['time_in_hospital_log'] = np.log1p(input_data['time_in_hospital'])
        feature_dict['number_outpatient_log'] = np.log1p(input_data['number_outpatient'])
        feature_dict['number_emergency_log'] = np.log1p(input_data['number_emergency'])
        feature_dict['number_inpatient_log'] = np.log1p(input_data['number_inpatient'])
        feature_dict['num_medications_log'] = np.log1p(input_data['num_medications'])
        
        # Diagnosis categories - set meaningful defaults based on patient characteristics
        # More diagnoses = higher category
        if input_data['number_diagnoses'] >= 10:
            category = 8  # High complexity
        elif input_data['number_diagnoses'] >= 7:
            category = 4  # Moderate complexity
        else:
            category = 0  # Low complexity
        
        feature_dict['diag_1_category'] = category
        feature_dict['diag_2_category'] = category
        feature_dict['diag_3_category'] = category
        
        features_df = pd.DataFrame([feature_dict])
        missing_features = set(self.feature_names) - set(features_df.columns)
        for feat in missing_features:
            features_df[feat] = 0
        features_df = features_df[self.feature_names]
        
        # Convert to numpy array
        features_array = features_df.values.astype(float)
        
        return features_array
    
    def make_prediction(self, input_data):
        if self.model is None:
            return None, None
        try:
            features = self.preprocess_input(input_data)
            
            # Scale features using the scaler
            if self.scaler is not None:
                features_scaled = self.scaler.transform(features)
            else:
                features_scaled = features
            
            prediction = self.model.predict(features_scaled)[0]
            probability = self.model.predict_proba(features_scaled)[0]
            return prediction, probability
        except Exception as e:
            st.error(f"Prediction error: {e}")
            return None, None
    
    def display_prediction_results(self, prediction, probability):
        if prediction is None:
            return
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("### 🎯 Prediction")
            if prediction == 1:
                st.markdown('<p class="risk-high">🔴 HIGH RISK - Readmission Likely</p>', unsafe_allow_html=True)
            else:
                st.markdown('<p class="risk-low">🟢 LOW RISK - No Readmission Expected</p>', unsafe_allow_html=True)
        
        with col2:
            st.markdown("### 📊 Confidence")
            risk_probability = probability[1] * 100
            st.metric("Risk Probability", f"{risk_probability:.1f}%")
        
        with col3:
            st.markdown("### ⚠️ Risk Level")
            if risk_probability >= 70:
                risk_level = "🔴 HIGH"
                risk_class = "risk-high"
            elif risk_probability >= 40:
                risk_level = "🟡 MEDIUM"
                risk_class = "risk-medium"
            else:
                risk_level = "🟢 LOW"
                risk_class = "risk-low"
            st.markdown(f'<p class="{risk_class}">{risk_level}</p>', unsafe_allow_html=True)
    
    def create_risk_visualization(self, probability):
        if probability is None:
            return
        
        risk_prob = probability[1] * 100
        
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = risk_prob,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Readmission Risk (%)"},
            delta = {'reference': 50},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 30], 'color': "lightgreen"},
                    {'range': [30, 70], 'color': "yellow"},
                    {'range': [70, 100], 'color': "red"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 70
                }
            }
        ))
        
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    def display_risk_factors(self, input_data):
        """Display key risk factors explaining the prediction."""
        
        risk_prob = st.session_state.probability[1] if 'probability' in st.session_state else 0
        is_high_risk = risk_prob >= 0.7
        is_medium_risk = 0.4 <= risk_prob < 0.7
        
        risk_factors = []
        
        # Age factor
        if input_data['age'] >= 75:
            risk_factors.append("🔴 **Advanced Age (75+)**: Elderly patients have slower recovery and higher vulnerability")
        elif input_data['age'] >= 65:
            risk_factors.append("🟡 **Age (65+)**: Moderate age-related risk factors")
        else:
            risk_factors.append("🟢 **Age (Under 65)**: Lower age-related risk")
        
        # Hospital stay
        if input_data['time_in_hospital'] >= 8:
            risk_factors.append("🔴 **Extended Hospital Stay (8+ days)**: Indicates complex medical condition requiring prolonged care")
        elif input_data['time_in_hospital'] >= 5:
            risk_factors.append("🟡 **Moderate Stay (5-7 days)**: Some complexity in case management")
        else:
            risk_factors.append("🟢 **Short Stay (<5 days)**: Less complex case, faster recovery")
        
        # Previous admissions
        if input_data['number_inpatient'] >= 3:
            risk_factors.append(f"🔴 **Previous Inpatient Stays ({input_data['number_inpatient']})**: Established pattern of readmission, indicates chronic or recurring conditions")
        elif input_data['number_inpatient'] >= 1:
            risk_factors.append(f"🟡 **Previous Admissions ({input_data['number_inpatient']})**: Some prior readmission history")
        else:
            risk_factors.append("🟢 **No Previous Admissions**: No established readmission pattern")
        
        # Emergency visits
        if input_data['number_emergency'] >= 2:
            risk_factors.append(f"🔴 **Emergency Visits ({input_data['number_emergency']})**: Recent urgent health episodes indicate instability")
        elif input_data['number_emergency'] >= 1:
            risk_factors.append(f"🟡 **Emergency Visits ({input_data['number_emergency']})**: Some recent emergency care")
        else:
            risk_factors.append("🟢 **No Emergency Visits**: Stable care pattern")
        
        # Diagnoses
        if input_data['number_diagnoses'] >= 10:
            risk_factors.append(f"🔴 **Complex Conditions ({input_data['number_diagnoses']} diagnoses)**: High medical complexity, multiple systems affected")
        elif input_data['number_diagnoses'] >= 7:
            risk_factors.append(f"🟡 **Multiple Conditions ({input_data['number_diagnoses']} diagnoses)**: Moderate medical complexity")
        else:
            risk_factors.append(f"🟢 **Few Conditions ({input_data['number_diagnoses']} diagnoses)**: Lower medical complexity")
        
        # Medications
        if input_data['num_medications'] >= 20:
            risk_factors.append(f"🟡 **High Medication Load ({input_data['num_medications']})**: Complex medication management needed")
        elif input_data['num_medications'] >= 10:
            risk_factors.append(f"🟢 **Moderate Medications ({input_data['num_medications']})**: Manageable medication regime")
        else:
            risk_factors.append(f"🟢 **Few Medications ({input_data['num_medications']})**: Simple medication management")
        
        # Show the factors
        for factor in risk_factors:
            st.markdown(f"- {factor}")
        
        # Summary
        if is_high_risk:
            st.info("⚠️ **Summary**: Multiple high-risk factors indicate need for enhanced discharge planning and close follow-up")
        elif is_medium_risk:
            st.warning("📊 **Summary**: Moderate risk factors present - standard discharge planning with scheduled follow-up recommended")
        else:
            st.success("✅ **Summary**: Lower risk profile - patient is stable with good post-discharge outlook")
    
    def run(self):
        self.create_header()
        
        tab1, tab2, tab3 = st.tabs(["🔮 Prediction", "📊 Analytics", "ℹ️ About"])
        
        with tab1:
            col1, col2 = st.columns([1, 1.2])
            
            with col1:
                input_data = self.create_sidebar()
                
                if st.button("🔍 Predict Readmission Risk", type="primary", use_container_width=True):
                    with st.spinner("Analyzing patient data..."):
                        prediction, probability = self.make_prediction(input_data)
                        
                        if prediction is not None:
                            st.success("✅ Prediction completed!")
                            st.session_state.prediction = prediction
                            st.session_state.probability = probability
            
            with col2:
                if 'prediction' in st.session_state:
                    self.display_prediction_results(st.session_state.prediction, st.session_state.probability)
                    st.subheader("🎯 Risk Assessment")
                    self.create_risk_visualization(st.session_state.probability)
                    
                    # Show reasons for prediction
                    st.subheader("🔍 Key Risk Factors")
                    self.display_risk_factors(input_data)
                else:
                    st.info("👈 Enter patient data and click 'Predict' to see results")
        
        with tab2:
            st.header("📈 Model Analytics")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Dataset Statistics")
                if os.path.exists('data/X_train.csv'):
                    train_data = pd.read_csv('data/X_train.csv')
                    st.write(f"**Training Samples:** {len(train_data):,}")
                    st.write(f"**Features:** {len(train_data.columns)}")
                    
                    if os.path.exists('data/y_train.csv'):
                        y_train = pd.read_csv('data/y_train.csv')
                        readmission_rate = y_train.mean()[0] * 100
                        st.write(f"**Readmission Rate:** {readmission_rate:.1f}%")
            
            with col2:
                st.subheader("Model Performance")
                if os.path.exists('models/training_results.joblib'):
                    results = joblib.load('models/training_results.joblib')
                    if results:
                        best_model = max(results.keys(), key=lambda x: results[x]['best_score'])
                        best_score = results[best_model]['best_score']
                        st.write(f"**Best Model:** {best_model}")
                        st.write(f"**Cross-validation AUC:** {best_score:.3f}")
                        st.write(f"**Test Accuracy:** 71%")
                        st.write(f"**Test AUC:** 0.66")
        
        with tab3:
            st.header("About This Application")
            st.markdown("""
            ### 🏥 Hospital Readmission Risk Predictor
            
            AI-powered application for predicting patient readmission risk using machine learning.
            
            **Features:**
            - Real-time risk assessment
            - Interactive visualizations
            - Batch prediction support
            - Model explainability
            
            **Technology:**
            - Python, Scikit-learn, Random Forest
            - Streamlit for web interface
            - Plotly for visualizations
            
            **Performance:**
            - 71% accuracy
            - 0.66 AUC score
            - 96 engineered features
            """)

def main():
    dashboard = HospitalReadmissionDashboard()
    dashboard.run()

if __name__ == "__main__":
    main()