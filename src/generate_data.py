import pandas as pd
import numpy as np
import os

def generar_datos_sinteticos(n_muestras=200):
    os.makedirs('data', exist_ok=True)
    np.random.seed(42)
    
    data = {
        'Edad': np.random.randint(18, 75, n_muestras),
        'Genero': np.random.choice(['Hombre', 'Mujer'], n_muestras),
        'Ingresos': np.random.randint(1500000, 15000000, n_muestras), # Pesos/Moneda local
        'Educacion': np.random.choice(['Bachillerato', 'Pregrado', 'Maestría', 'Doctorado'], n_muestras),
        'Estado_Civil': np.random.choice(['Soltero', 'Casado', 'Divorciado'], n_muestras),
        'Hijos': np.random.randint(0, 5, n_muestras),
        'Vivienda': np.random.choice(['Renta', 'Propia', 'Hipoteca'], n_muestras),
        'Puntaje_Crediticio': np.random.choice(['Bajo', 'Promedio', 'Alto'], n_muestras, p=[0.2, 0.5, 0.3])
    }
    
    df = pd.DataFrame(data)
    df.to_csv('data/synthetic_data.csv', index=False)
    print("[CAPA DE DATOS] >> Datos localizados generados en data/synthetic_data.csv")

if __name__ == "__main__":
    generar_datos_sinteticos()
