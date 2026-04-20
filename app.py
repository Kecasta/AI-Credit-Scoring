import streamlit as st
import joblib
import pandas as pd
import os
import plotly.express as px

# Configuración de Marca y Estilo
st.set_page_config(page_title="Scoring Crediticio IA", page_icon="💳", layout="wide")

# Estilo personalizado para KPI Cards
st.markdown("""
<style>
    .kpi-card {
        background-color: #1E2227;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #3498DB;
        margin-bottom: 20px;
    }
    .kpi-value {
        font-size: 24px;
        font-weight: bold;
        color: #2ECC71;
    }
    .kpi-label {
        font-size: 14px;
        color: #A0A0A0;
    }
</style>
""", unsafe_allow_html=True)

# Carga de Inteligencia
@st.cache_resource
def load_model_assets():
    if os.path.exists('models/model.pkl') and os.path.exists('models/mappings.joblib'):
        modelo = joblib.load('models/model.pkl')
        mappings = joblib.load('models/mappings.joblib')
        return modelo, mappings
    return None, None

modelo, mappings = load_model_assets()

# --- NAVEGACIÓN ---
tab1, tab2 = st.tabs(["📊 Dashboard Analítico", "🧮 Calculadora de Riesgo"])

# --- TAB 1: DASHBOARD ---
with tab1:
    st.title("📊 Dashboard de Crédito y Cartera")
    st.markdown("---")
    
    if os.path.exists('data/synthetic_data.csv'):
        df = pd.read_csv('data/synthetic_data.csv')
        
        # KPIs Superiores
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown('<div class="kpi-card"><div class="kpi-label">Total Solicitudes</div><div class="kpi-value">{}</div></div>'.format(len(df)), unsafe_allow_html=True)
        with col2:
            avg_income = df['Ingresos'].mean()
            st.markdown('<div class="kpi-card"><div class="kpi-label">Ingreso Promedio</div><div class="kpi-value">${:,.0f}</div></div>'.format(avg_income), unsafe_allow_html=True)
        with col3:
            approved_pct = (df['Puntaje_Crediticio'] == 'Alto').mean() * 100
            st.markdown('<div class="kpi-card"><div class="kpi-label">% Aprobación (Alto)</div><div class="kpi-value">{:.1f}%</div></div>'.format(approved_pct), unsafe_allow_html=True)
        with col4:
            most_common_edu = df['Educacion'].mode()[0]
            st.markdown('<div class="kpi-card"><div class="kpi-label">Educación Predominante</div><div class="kpi-value">{}</div></div>'.format(most_common_edu), unsafe_allow_html=True)
            
        st.markdown("### Visualización de Riesgo y Perfiles")
        
        c1, c2 = st.columns(2)
        
        with c1:
            # Gráfico de Barras: Distribución de Riesgos
            fig_bar = px.histogram(df, x="Puntaje_Crediticio", 
                                  title="Distribución de Resultados de Scoring",
                                  color="Puntaje_Crediticio",
                                  color_discrete_map={"Bajo": "#E74C3C", "Promedio": "#F1C40F", "Alto": "#2ECC71"},
                                  category_orders={"Puntaje_Crediticio": ["Bajo", "Promedio", "Alto"]})
            st.plotly_chart(fig_bar, use_container_width=True)
            
        with c2:
            # Gráfico de Torta: Educación
            fig_pie = px.pie(df, names="Educacion", title="Distribución por Nivel Educativo",
                            color_discrete_sequence=px.colors.qualitative.Pastel)
            st.plotly_chart(fig_pie, use_container_width=True)
            
        # Scatter Plot: Edad vs Ingresos
        st.markdown("### Correlación Edad vs Ingresos")
        fig_scatter = px.scatter(df, x="Edad", y="Ingresos", color="Puntaje_Crediticio",
                                 title="Análisis de Ingresos por Edad y Riesgo",
                                 color_discrete_map={"Bajo": "#E74C3C", "Promedio": "#F1C40F", "Alto": "#2ECC71"},
                                 hover_data=['Estado_Civil', 'Vivienda'],
                                 trendline="ols" if len(df) > 1 else None)
        st.plotly_chart(fig_scatter, use_container_width=True)
        
    else:
        st.warning("No se encontró el archivo de datos. Por favor, genera los datos primero.")

# --- TAB 2: CALCULADORA ---
with tab2:
    st.title("💳 Evaluador de Riesgo Crediticio")
    st.markdown("---")

    if modelo is not None and mappings is not None:
        # Sidebar Localizado
        with st.sidebar:
            st.header("📋 Perfil del Solicitante")
            st.info("Ajuste los parámetros para evaluar el riesgo en tiempo real.")
            edad = st.slider("Edad", 18, 75, 30)
            genero = st.selectbox("Género", list(mappings['Genero'].keys()))
            ingresos = st.number_input("Ingresos Mensuales", min_value=0, value=3000000, step=100000)
            educacion = st.selectbox("Nivel Educativo", list(mappings['Educacion'].keys()))
            estado_civil = st.selectbox("Estado Civil", list(mappings['Estado_Civil'].keys()))
            hijos = st.number_input("Número de Hijos", 0, 10, 0)
            vivienda = st.selectbox("Tipo de Vivienda", list(mappings['Vivienda'].keys()))

        # Lógica de Predicción
        if st.button("🚀 Iniciar Evaluación"):
            entrada = pd.DataFrame([{
                'Edad': edad, 'Genero': genero, 'Ingresos': ingresos,
                'Educacion': educacion, 'Estado_Civil': estado_civil,
                'Hijos': hijos, 'Vivienda': vivienda
            }])
            
            for col, mapping in mappings.items():
                if col in entrada.columns:
                    entrada[col] = entrada[col].map(mapping)
            
            entrada = entrada[['Edad', 'Genero', 'Ingresos', 'Educacion', 'Estado_Civil', 'Hijos', 'Vivienda']]
            
            prediccion = modelo.predict(entrada)[0]
            
            st.subheader("Dictamen Final:")
            if prediccion == 0: # Bajo
                st.error("🔴 **RIESGO DE CRÉDITO ALTO**")
                st.write("La solicitud presenta indicadores de alto riesgo de incumplimiento.")
            elif prediccion == 1: # Promedio
                st.warning("🟡 **RIESGO DE CRÉDITO MODERADO**")
                st.write("Se recomienda un análisis manual adicional de las garantías.")
            else: # Alto
                st.success("🟢 **CRÉDITO APROBADO (RIESGO BAJO)**")
                st.write("El perfil cumple con los estándares óptimos de solvencia.")
    else:
        st.error("🚨 Modelo no detectado. Re-entrene el sistema.")

st.markdown("---")
st.caption("Nexus AI Credit System | Dashboard Analítico V1.0")
