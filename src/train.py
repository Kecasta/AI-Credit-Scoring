import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
import os

def train_model():
    os.makedirs('models', exist_ok=True)
    df = pd.read_csv('data/synthetic_data.csv')
    
    # Mapping Encoders (Vibe Coding Protocol)
    mappings = {
        'Gender': {'Male': 0, 'Female': 1},
        'Education': {'High School': 0, 'Bachelor': 1, 'Master': 2, 'Doctorate': 3},
        'Marital Status': {'Single': 0, 'Married': 1, 'Divorced': 2},
        'Home Ownership': {'Rented': 0, 'Owned': 1, 'Mortgage': 2},
        'Credit Score': {'Low': 0, 'Average': 1, 'High': 2}
    }
    
    for col, mapping in mappings.items():
        df[col] = df[col].map(mapping)
    
    X = df.drop('Credit Score', axis=1)
    y = df['Credit Score']
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    joblib.dump(model, 'models/model.pkl')
    joblib.dump(mappings, 'models/mappings.joblib')
    print("[ML LAYER] >> Modelo y Mappings exportados a models/")

if __name__ == "__main__":
    train_model()
