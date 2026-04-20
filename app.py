import streamlit as st
import joblib
import pandas as pd
import os
import plotly.express as px

# Configuración de Marca y Estilo
st.set_page_config(page_title="Scoring Crediticio IA", page_icon="💳", layout="centered")

# Carga de Inteligencia
@st.cache_resource
def load_model_assets():
    if os.path.exists('models/model.pkl') and os.path.exists('models/mappings.joblib'):
        modelo = joblib.load('models/model.pkl')
        mappings = joblib.load('models/mappings.joblib')
        return modelo, mappings
    return None, None

modelo, mappings = load_model_assets()

st.title("💳 Evaluador de Riesgo Crediticio")
st.markdown("---")

if modelo is not None and mappings is not None:
    # Sidebar Localizado
    with st.sidebar:
        st.header("📋 Información del Solicitante")
        edad = st.slider("Edad", 18, 75, 30)
        genero = st.selectbox("Género", list(mappings['Genero'].keys()))
        ingresos = st.number_input("Ingresos Mensuales", min_value=0, value=3000000, step=100000)
        educacion = st.selectbox("Nivel Educativo", list(mappings['Educacion'].keys()))
        estado_civil = st.selectbox("Estado Civil", list(mappings['Estado_Civil'].keys()))
        hijos = st.number_input("Número de Hijos", 0, 10, 0)
        vivienda = st.selectbox("Tipo de Vivienda", list(mappings['Vivienda'].keys()))

    # Lógica de Predicción
    if st.button("🚀 Evaluar Solicitud"):
        # Construcción del dataframe de entrada
        entrada = pd.DataFrame([{
            'Edad': edad, 'Genero': genero, 'Ingresos': ingresos,
            'Educacion': educacion, 'Estado_Civil': estado_civil,
            'Hijos': hijos, 'Vivienda': vivienda
        }])
        
        # Aplicar Mapeos
        for col, mapping in mappings.items():
            if col in entrada.columns:
                entrada[col] = entrada[col].map(mapping)
        
        # Reordenar columnas para coincidir con el entrenamiento (X)
        entrada = entrada[['Edad', 'Genero', 'Ingresos', 'Educacion', 'Estado_Civil', 'Hijos', 'Vivienda']]
        
        prediccion = modelo.predict(entrada)[0]
        
        # Sistema de Alertas Visuales (Semáforo)
        st.subheader("Resultado de la Evaluación:")
        if prediccion == 0: # Bajo
            st.error("🔴 **RIESGO DE CRÉDITO ALTO**")
            st.write("La solicitud presenta indicadores de alto riesgo de incumplimiento.")
        elif prediccion == 1: # Promedio
            st.warning("🟡 **RIESGO DE CRÉDITO MODERADO**")
            st.write("Se recomienda un análisis manual adicional de las garantías.")
        else: # Alto (Buen puntaje)
            st.success("🟢 **CRÉDITO APROBADO (RIESGO BAJO)**")
            st.write("El perfil cumple con los estándares óptimos de solvencia.")

        st.markdown("---")
        st.subheader("📊 Análisis de Factores Determinantes")
        
        # 2. KPIs rápidos
        col_kpi1, col_kpi2, col_kpi3 = st.columns(3)
        probabilidades = modelo.predict_proba(entrada)[0]
        
        with col_kpi1:
            st.metric("Confianza de IA", f"{max(probabilidades)*100:.1f}%")
        with col_kpi2:
            st.metric("Ingresos Evaluados", f"${ingresos:,.0f}")
        with col_kpi3:
            st.metric("Estabilidad", "Alta" if edad > 30 and estado_civil != 'Soltero' else "Media")

        # 3. Gráfico de importancia (Por qué salió ese resultado)
        importancias = modelo.feature_importances_
        fig = px.bar(
            x=importancias, 
            y=entrada.columns, 
            orientation='h',
            title="Peso de las Variables en su Resultado",
            labels={'x': 'Importancia Relativa', 'y': 'Variable'},
            color_discrete_sequence=['#3498DB']
        )
        fig.update_layout(height=400, margin=dict(l=20, r=20, t=40, b=20), yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig, use_container_width=True)
else:
    st.error("🚨 ¡Cerebro de IA no encontrado! Por favor, ejecuta el entrenamiento.")
    if st.button("Re-entrenar Sistema"):
        os.system("python src/generate_data.py")
        os.system("python src/train.py")
        st.rerun()

st.markdown("---")
st.caption("Nexus AI Credit System | Interfaz Reactiva y Análisis Dinámico")
