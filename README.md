# 🚀 AI Credit Scoring: Sistema Experto con Inteligencia Agéntica

Esta plataforma es una solución de IA Híbrida diseñada para el sector Fintech. Combina el rigor matemático del Machine Learning Tradicional con la capacidad cognitiva de Agentes Autónomos (LLMs) para ofrecer una evaluación de riesgo crediticio con explicabilidad humana en tiempo real.

## 🌟 Lo que hace único a este proyecto

A diferencia de los evaluadores estándar, este sistema utiliza una **Arquitectura de Dos Capas**:

- **Capa Predictiva (El Ojo):** Un modelo `RandomForestClassifier` que garantiza precisión técnica y métricas deterministas.
- **Capa Agéntica (La Voz):** Un Agente de IA potenciado por **Llama 3.3** (vía Groq) que actúa como un analista senior, interpretando el resultado y redactando un plan de acción financiero personalizado para el solicitante.

## 🛠️ Stack Tecnológico

- **Core:** Python 3.10+
- **ML Engine:** Scikit-Learn (Random Forest)
- **AI Orchestration:** LangChain (Agentic Workflows)
- **LLM Infrastructure:** Groq Cloud (Llama 3.3 Versatile)
- **Interface:** Streamlit (Premium UX/UI)
- **Data Ops:** Pandas, Joblib, Python-Dotenv

## 📁 Estructura de la Solución (Modular)

```plaintext
├── models/               # Artefactos del modelo (.pkl) y mapeos
├── src/
│   ├── agents/          # [NUEVO] Capa de razonamiento y prompts del Agente
│   ├── generate_data.py # Simulación de datos sintéticos controlados
│   └── train.py         # Pipeline de entrenamiento y exportación
├── app.py               # Dashboard interactivo y orquestación final
└── requirements.txt     # Dependencias del ecosistema
```

## 📦 Instalación y Configuración

**Clonar y Preparar:**
```bash
git clone https://github.com/Kecasta/AI-Credit-Scoring.git
cd AI-Credit-Scoring
```

**Variables de Entorno:**
Crea un archivo `.env` en la raíz y añade tu llave de Groq:
```env
GROQ_API_KEY="tu_api_key_aqui"
```

**Despliegue Local:**
```bash
pip install -r requirements.txt
streamlit run app.py
```

## 💼 Impacto de Negocio & Casos de Uso

- **Optimización Operativa:** Reduce el tiempo de respuesta de analistas de crédito de horas a milisegundos.
- **Fidelización del Cliente:** No solo entrega un "No", entrega un plan de mejora financiera redactado por IA.
- **Escalabilidad:** Arquitectura lista para integrarse con APIs bancarias reales y flujos de trabajo en la nube.

---
**Desarrollado por Kevin Seryeit Castañeda Aldana**  
*Ingeniero de Sistemas | Especialista en Soluciones de IA Agéntica & Backend*
