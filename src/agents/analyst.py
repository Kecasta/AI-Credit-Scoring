import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from .prompts import SYSTEM_PROMPT

load_dotenv()

class CreditAnalyst:
    def __init__(self):
        # Usamos gpt-4o-mini por eficiencia y costo
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
        
    def generar_informe(self, score_resultado, datos_usuario):
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("user", "Resultado IA: {resultado}. Perfil del Cliente: {perfil}")
        ])
        
        # Formatear la cadena de entrada
        cadena = prompt_template | self.llm
        
        perfil_str = f"Edad: {datos_usuario['Edad']}, Ingresos: {datos_usuario['Ingresos']}, Educación: {datos_usuario['Educacion']}"
        
        respuesta = cadena.invoke({
            "resultado": score_resultado,
            "perfil": perfil_str
        })
        
        return respuesta.content
