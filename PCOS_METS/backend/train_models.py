import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os
from datetime import datetime

print("🚀 Starting ML Model Training for Health Screening System...")
print("=" * 60)

# Create models directory if it doesn't exist
os.makedirs('../models', exist_ok=True)

# ==============================================
# PCOS MODEL TRAINING
# ==============================================

print("\n📋 Training PCOS Model...")

# Load PCOS dataset (simulated - replace with actual Kaggle dataset)
# For now, creating a realistic PCOS dataset with obesity-related features
np.random.seed(42)
n_samples = 1000

pcos_data = {
    'age': np.random.randint(18, 45, n_samples),
    'weight': np.random.normal(70, 15, n_samples),
    'height': np.random.normal(162, 8, n_samples),
    'waist_circumference': np.random.normal(85, 12, n_samples),
    'hip_circumference': np.random.normal(95, 10, n_samples),
    'body_fat_percentage': np.random.normal(28, 8, n_samples),
    'visceral_fat_index': np.random.normal(10, 4, n_samples),
    'diet_calories': np.random.normal(2200, 400, n_samples),
    'fat_intake_grams': np.random.normal(80, 20, n_samples),
    'carb_intake_grams': np.random.normal(275, 50, n_samples),
    'protein_intake_grams': np.random.normal(90, 15, n_samples),
    'physical_activity_hours': np.random.normal(1.5, 1.0, n_samples),
    'sedentary_hours': np.random.normal(8, 3, n_samples),
    'fast_food_frequency': np.random.normal(4, 3, n_samples),
    'sugar_beverage_frequency': np.random.normal(2, 2, n_samples),
    'insulin_resistance_score': np.random.normal(4, 2, n_samples),
    'leptin_level': np.random.normal(15, 8, n_samples),
    'adiponectin_level': np.random.normal(8, 3, n_samples),
    'crp_level': np.random.normal(3.5, 2, n_samples)
}

pcos_df = pd.DataFrame(pcos_data)

# Calculate BMI
pcos_df['bmi'] = pcos_df['weight'] / ((pcos_df['height'] / 100) ** 2)

# Generate PCOS labels based on obesity-related risk factors
# Higher BMI, insulin resistance, and visceral fat increase PCOS risk
pcos_risk_score = (
    (pcos_df['bmi'] > 25).astype(int) * 0.3 +
    (pcos_df['insulin_resistance_score'] > 4).astype(int) * 0.25 +
    (pcos_df['visceral_fat_index'] > 10).astype(int) * 0.2 +
    (pcos_df['leptin_level'] > 12).astype(int) * 0.15 +
    (pcos_df['crp_level'] > 3).astype(int) * 0.1
)

pcos_df['pcos'] = (pcos_risk_score + np.random.normal(0, 0.1, n_samples) > 0.5).astype(int)

# Select obesity-related features for PCOS
pcos_features = [
    'age', 'weight', 'height', 'bmi', 'waist_circumference', 'hip_circumference',
    'body_fat_percentage', 'visceral_fat_index', 'diet_calories', 'fat_intake_grams',
    'carb_intake_grams', 'protein_intake_grams', 'physical_activity_hours',
    'sedentary_hours', 'fast_food_frequency', 'sugar_beverage_frequency',
    'insulin_resistance_score', 'leptin_level', 'adiponectin_level', 'crp_level'
]

pcos_X = pcos_df[pcos_features]
pcos_y = pcos_df['pcos']

# Split PCOS data
pcos_X_train, pcos_X_test, pcos_y_train, pcos_y_test = train_test_split(
    pcos_X, pcos_y, test_size=0.2, random_state=42, stratify=pcos_y
)

# Scale PCOS features
pcos_scaler = StandardScaler()
pcos_X_train_scaled = pcos_scaler.fit_transform(pcos_X_train)
pcos_X_test_scaled = pcos_scaler.transform(pcos_X_test)

# Train PCOS XGBoost model
pcos_model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'
)

pcos_model.fit(pcos_X_train_scaled, pcos_y_train)

# Evaluate PCOS model
pcos_y_pred = pcos_model.predict(pcos_X_test_scaled)
pcos_accuracy = accuracy_score(pcos_y_test, pcos_y_pred)

print(f"✅ PCOS Model Accuracy: {pcos_accuracy:.3f}")
print(f"📊 PCOS Test Distribution: {pcos_y_test.value_counts().to_dict()}")

# Save PCOS model and scaler
joblib.dump(pcos_model, '../models/pcos_xgboost.pkl')
joblib.dump(pcos_scaler, '../models/pcos_scaler.pkl')
joblib.dump(pcos_features, '../models/pcos_features.pkl')

print("💾 PCOS model saved successfully!")

# ==============================================
# METABOLIC SYNDROME MODEL TRAINING
# ==============================================

print("\n📋 Training Metabolic Syndrome Model...")

# Load Metabolic Syndrome dataset (simulated - replace with actual Kaggle dataset)
n_samples_met = 1200

metabolic_data = {
    'age': np.random.randint(25, 70, n_samples_met),
    'gender': np.random.choice(['male', 'female'], n_samples_met),
    'weight': np.random.normal(75, 18, n_samples_met),
    'height': np.random.normal(170, 10, n_samples_met),
    'waist_circumference': np.random.normal(95, 15, n_samples_met),
    'hip_circumference': np.random.normal(100, 12, n_samples_met),
    'body_fat_percentage': np.random.normal(30, 10, n_samples_met),
    'visceral_fat_index': np.random.normal(12, 5, n_samples_met),
    'diet_calories': np.random.normal(2400, 500, n_samples_met),
    'fat_intake_grams': np.random.normal(90, 25, n_samples_met),
    'carb_intake_grams': np.random.normal(300, 60, n_samples_met),
    'protein_intake_grams': np.random.normal(100, 20, n_samples_met),
    'physical_activity_hours': np.random.normal(1.2, 1.2, n_samples_met),
    'sedentary_hours': np.random.normal(9, 3, n_samples_met),
    'fast_food_frequency': np.random.normal(5, 4, n_samples_met),
    'sugar_beverage_frequency': np.random.normal(3, 2, n_samples_met),
    'insulin_resistance_score': np.random.normal(5, 2.5, n_samples_met),
    'leptin_level': np.random.normal(18, 10, n_samples_met),
    'adiponectin_level': np.random.normal(7, 3, n_samples_met),
    'crp_level': np.random.normal(4, 2.5, n_samples_met),
    'uric_acid_level': np.random.normal(6.5, 2, n_samples_met),
    'liver_fat_percentage': np.random.normal(15, 8, n_samples_met)
}

metabolic_df = pd.DataFrame(metabolic_data)

# Calculate BMI
metabolic_df['bmi'] = metabolic_df['weight'] / ((metabolic_df['height'] / 100) ** 2)

# Encode gender
le = LabelEncoder()
metabolic_df['gender_encoded'] = le.fit_transform(metabolic_df['gender'])

# Generate Metabolic Syndrome labels based on obesity-related risk factors
# Higher BMI, waist circumference, and metabolic markers increase risk
metabolic_risk_score = (
    (metabolic_df['bmi'] > 27).astype(int) * 0.25 +
    ((metabolic_df['waist_circumference'] > 102) & (metabolic_df['gender'] == 'male')).astype(int) * 0.25 +
    ((metabolic_df['waist_circumference'] > 88) & (metabolic_df['gender'] == 'female')).astype(int) * 0.25 +
    (metabolic_df['insulin_resistance_score'] > 4).astype(int) * 0.2 +
    (metabolic_df['uric_acid_level'] > 7).astype(int) * 0.15 +
    (metabolic_df['liver_fat_percentage'] > 10).astype(int) * 0.15
)

metabolic_df['metabolic_syndrome'] = (metabolic_risk_score + np.random.normal(0, 0.1, n_samples_met) > 0.5).astype(int)

# Select obesity-related features for Metabolic Syndrome
metabolic_features = [
    'age', 'gender_encoded', 'weight', 'height', 'bmi', 'waist_circumference', 
    'hip_circumference', 'body_fat_percentage', 'visceral_fat_index', 
    'diet_calories', 'fat_intake_grams', 'carb_intake_grams', 'protein_intake_grams',
    'physical_activity_hours', 'sedentary_hours', 'fast_food_frequency',
    'sugar_beverage_frequency', 'insulin_resistance_score', 'leptin_level',
    'adiponectin_level', 'crp_level', 'uric_acid_level', 'liver_fat_percentage'
]

metabolic_X = metabolic_df[metabolic_features]
metabolic_y = metabolic_df['metabolic_syndrome']

# Split Metabolic data
metabolic_X_train, metabolic_X_test, metabolic_y_train, metabolic_y_test = train_test_split(
    metabolic_X, metabolic_y, test_size=0.2, random_state=42, stratify=metabolic_y
)

# Scale Metabolic features
metabolic_scaler = StandardScaler()
metabolic_X_train_scaled = metabolic_scaler.fit_transform(metabolic_X_train)
metabolic_X_test_scaled = metabolic_scaler.transform(metabolic_X_test)

# Train Metabolic Syndrome XGBoost model
metabolic_model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'
)

metabolic_model.fit(metabolic_X_train_scaled, metabolic_y_train)

# Evaluate Metabolic model
metabolic_y_pred = metabolic_model.predict(metabolic_X_test_scaled)
metabolic_accuracy = accuracy_score(metabolic_y_test, metabolic_y_pred)

print(f"✅ Metabolic Syndrome Model Accuracy: {metabolic_accuracy:.3f}")
print(f"📊 Metabolic Test Distribution: {metabolic_y_test.value_counts().to_dict()}")

# Save Metabolic model and scaler
joblib.dump(metabolic_model, '../models/metabolic_xgboost.pkl')
joblib.dump(metabolic_scaler, '../models/metabolic_scaler.pkl')
joblib.dump(metabolic_features, '../models/metabolic_features.pkl')

print("💾 Metabolic Syndrome model saved successfully!")

# ==============================================
# MODEL SUMMARY
# ==============================================

print("\n" + "=" * 60)
print("📈 MODEL TRAINING SUMMARY")
print("=" * 60)

print(f"✅ PCOS Model:")
print(f"   - Features: {len(pcos_features)}")
print(f"   - Accuracy: {pcos_accuracy:.3f}")
print(f"   - Training samples: {len(pcos_X_train)}")
print(f"   - Test samples: {len(pcos_X_test)}")

print(f"\n✅ Metabolic Syndrome Model:")
print(f"   - Features: {len(metabolic_features)}")
print(f"   - Accuracy: {metabolic_accuracy:.3f}")
print(f"   - Training samples: {len(metabolic_X_train)}")
print(f"   - Test samples: {len(metabolic_X_test)}")

print(f"\n🎯 Feature Importance (PCOS Top 5):")
feature_importance_pcos = pcos_model.feature_importances_
top_features_pcos = sorted(zip(pcos_features, feature_importance_pcos), key=lambda x: x[1], reverse=True)[:5]
for feature, importance in top_features_pcos:
    print(f"   - {feature}: {importance:.3f}")

print(f"\n🎯 Feature Importance (Metabolic Top 5):")
feature_importance_metabolic = metabolic_model.feature_importances_
top_features_metabolic = sorted(zip(metabolic_features, feature_importance_metabolic), key=lambda x: x[1], reverse=True)[:5]
for feature, importance in top_features_metabolic:
    print(f"   - {feature}: {importance:.3f}")

print(f"\n💾 Models saved in ../models/ directory:")
print(f"   - pcos_xgboost.pkl")
print(f"   - pcos_scaler.pkl")
print(f"   - pcos_features.pkl")
print(f"   - metabolic_xgboost.pkl")
print(f"   - metabolic_scaler.pkl")
print(f"   - metabolic_features.pkl")

print(f"\n🚀 Model training completed successfully!")
print(f"⏰ Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
