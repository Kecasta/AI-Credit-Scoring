import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from .prompts import SYSTEM_PROMPT

load_dotenv()

class CreditAnalyst:
    def __init__(self):
        # Usamos Llama 3 que es gratuito en la capa de Groq
        self.llm = ChatGroq(
            temperature=0, 
            model_name="llama3-70b-8192", 
            groq_api_key=os.getenv("GROQ_API_KEY")
        )
        
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
