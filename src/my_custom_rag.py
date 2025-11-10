# src/my_custom_rag.py (CORREGIDO Y FINAL)
import sys
import os
import time
from typing import List, Dict, Any, Optional, Tuple

from src.rag_interface import RAGSystem, RAGResult

# --- Importar el paquete RAG externo directamente ---
try:
    import rag_sandbox_package # <--- Importamos el paquete completo
    print("✅ Paquete 'rag_sandbox_package' del RAG externo importado con éxito.")
except ImportError as e:
    print(f"❌ ERROR: No se pudo importar el paquete 'rag_sandbox_package'.")
    print(f"Detalle: {e}")
    print("Asegúrate de que el RAG externo está correctamente configurado como un paquete y instalado con:")
    print(f"pip install -e /home/master/workspace/rag_sandbox")
    sys.exit(1)

class ExternalRAGAdapter(RAGSystem):
    def __init__(self):
        print("🌟 Inicializando tu RAG existente (llamando a rag_sandbox_package.rag_entrypoint.initialize_application())...")
        
        # Llama a la función de inicialización, que modificará las variables globales dentro de rag_sandbox_package.rag_entrypoint
        rag_sandbox_package.rag_entrypoint.initialize_application()

        # Accedemos a las variables globales *después* de la inicialización,
        # y las leemos DIRECTAMENTE desde el objeto módulo rag_sandbox_package.rag_entrypoint.
        self._rag_core_instance = rag_sandbox_package.rag_entrypoint.rag_core_instance
        self._llm_display_name = rag_sandbox_package.rag_entrypoint.llm_display_name

        if self._rag_core_instance is None:
            raise ValueError("La instancia de RAGCore no se inicializó correctamente en el RAG externo.")
        
        print("✅ Tu RAG existente inicializado correctamente a través del adaptador.")

    @property
    def llm_display_name(self) -> str:
        # Accedemos directamente a la variable global desde el submódulo rag_entrypoint
        return rag_sandbox_package.rag_entrypoint.llm_display_name

    def query(self, question: str) -> RAGResult:
        if self._rag_core_instance is None:
            raise RuntimeError("El adaptador RAG externo no ha sido inicializado.")

        active_filters = rag_sandbox_package.rag_entrypoint.active_pdf_filters
        
        rag_output_dict = self._rag_core_instance.process_query(
            user_query=question, # <--- La pregunta original
            active_pdf_filters=active_filters
        )

        # Añade la clave 'question' al diccionario de retorno
        return {
            "question": question,  
            "answer": rag_output_dict["answer"],
            "contexts": rag_output_dict["contexts"],
            "latency": rag_output_dict["latency"],
            "answer_tokens": rag_output_dict["answer_tokens"]
        }