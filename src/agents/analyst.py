import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from .prompts import SYSTEM_PROMPT

# 1. Cargamos las variables del archivo .env
load_dotenv()

class CreditAnalyst:
    def __init__(self):
        # 2. Configuramos el acceso a Groq usando la llave del .env
        # Usamos Llama 3 por su gran capacidad de razonamiento financiero
        self.llm = ChatGroq(
            temperature=0.7, 
            model_name="llama3-70b-8192", 
            groq_api_key=os.getenv("GROQ_API_KEY")
        )
        
    def generar_informe(self, score_resultado, datos_usuario):
        """
        Recibe el resultado del RandomForest y los datos del usuario 
        para generar un informe humano y profesional.
        """
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("user", "Resultado del Modelo IA: {resultado}. \nDatos del Solicitante: {perfil}")
        ])
        
        # Estructura de la cadena de razonamiento (Chain)
        cadena = prompt_template | self.llm
        
        # Formateamos los datos para el Agente
        perfil_str = (
            f"Edad: {datos_usuario['Edad']}, "
            f"Ingresos Mensuales: ${datos_usuario['Ingresos']}, "
            f"Nivel Educativo: {datos_usuario['Educacion']}"
        )
        
        # El Agente procesa y genera la respuesta
        respuesta = cadena.invoke({
            "resultado": "APROBADO" if score_resultado == 1 else "RECHAZADO",
            "perfil": perfil_str
        })
        
        return respuesta.content
