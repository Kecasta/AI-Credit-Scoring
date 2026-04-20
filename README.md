# 💳 AI Credit Scoring: Evaluador de Riesgo Inteligente

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-F7931E.svg)

Esta plataforma es una solución integral de **Machine Learning** diseñada para el sector Fintech. Permite evaluar el riesgo crediticio de solicitantes en tiempo real mediante un modelo predictivo de clasificación, ofreciendo una interfaz intuitiva y análisis de factores determinantes (XAI).

## 🚀 Características Principales

- **Modelo Predictivo:** Basado en `RandomForestClassifier` para una alta precisión y estabilidad.
- **Interpretabilidad (XAI):** Visualización dinámica de los factores que más influuyen en cada decisión crediticia.
- **Interfaz Premium:** Dashboard desarrollado en Streamlit con diseño enfocado en la experiencia de usuario (UX) y modo oscuro nativo.
- **Localización Total:** Sistema adaptado 100% al mercado de habla hispana.
- **KPIs en Tiempo Real:** Métricas de confianza de IA y estabilidad del solicitante tras cada evaluación.

## 🛠️ Arquitectura del Proyecto

El proyecto sigue una estructura modular para facilitar su escalabilidad y mantenimiento:

- `src/generate_data.py`: Script para la creación de datasets sintéticos controlados.
- `src/train.py`: Pipeline de entrenamiento y exportación de modelos.
- `app.py`: Aplicación web interactiva.
- `models/`: Artefactos del modelo y mapeos serializados.

## 📦 Instalación y Uso

1. **Clonar el repositorio:**
   ```bash
   git clone https://github.com/Kecasta/AI-Credit-Scoring.git
   cd AI-Credit-Scoring
   ```

2. **Instalar dependencias:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Generar datos y entrenar (opcional):**
   ```bash
   python src/generate_data.py
   python src/train.py
   ```

4. **Lanzar la aplicación:**
   ```bash
   streamlit run app.py
   ```

---
Desarrollado por **Kevin Seryeit Castañeda Aldana** - Ingeniero de Sistemas enfocado en soluciones de Inteligencia Artificial.
