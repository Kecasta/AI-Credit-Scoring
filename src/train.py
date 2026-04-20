import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
import os

def entrenar_modelo():
    os.makedirs('models', exist_ok=True)
    df = pd.read_csv('data/synthetic_data.csv')
    
    # Mapeos en Español (Vibe Coding Protocol)
    mappings = {
        'Genero': {'Hombre': 0, 'Mujer': 1},
        'Educacion': {'Bachillerato': 0, 'Pregrado': 1, 'Maestría': 2, 'Doctorado': 3},
        'Estado_Civil': {'Soltero': 0, 'Casado': 1, 'Divorciado': 2},
        'Vivienda': {'Renta': 0, 'Propia': 1, 'Hipoteca': 2},
        'Puntaje_Crediticio': {'Bajo': 0, 'Promedio': 1, 'Alto': 2}
    }
    
    for col, mapping in mappings.items():
        if col in df.columns:
            df[col] = df[col].map(mapping)
    
    X = df.drop('Puntaje_Crediticio', axis=1)
    y = df['Puntaje_Crediticio']
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    joblib.dump(model, 'models/model.pkl')
    joblib.dump(mappings, 'models/mappings.joblib')
    print("[ML LAYER] >> Inteligencia entrenada y localizada en models/")

if __name__ == "__main__":
    entrenar_modelo()
