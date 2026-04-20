import pandas as pd
import numpy as np
import os

def generate_synthetic_credit_data(n_samples=200):
    os.makedirs('data', exist_ok=True)
    np.random.seed(42)
    
    data = {
        'Age': np.random.randint(18, 70, n_samples),
        'Gender': np.random.choice(['Male', 'Female'], n_samples),
        'Income': np.random.randint(20000, 150000, n_samples),
        'Education': np.random.choice(['High School', 'Bachelor', 'Master', 'Doctorate'], n_samples),
        'Marital Status': np.random.choice(['Single', 'Married', 'Divorced'], n_samples),
        'Number of Children': np.random.randint(0, 5, n_samples),
        'Home Ownership': np.random.choice(['Rented', 'Owned', 'Mortgage'], n_samples),
        'Credit Score': np.random.choice(['Low', 'Average', 'High'], n_samples, p=[0.2, 0.5, 0.3])
    }
    
    df = pd.DataFrame(data)
    df.to_csv('data/synthetic_data.csv', index=False)
    print("[DATA LAYER] >> synthetic_data.csv generado exitosamente.")

if __name__ == "__main__":
    generate_synthetic_credit_data()
