# src/rag_interface.py
from typing import List, Dict, Protocol, Any, TypedDict, Optional

# Ya teníamos ProcessedChunk en rag_core.py, podemos re-definirlo aquí o importarlo si lo compartimos.
# Por simplicidad, re-definimos para que sea autocontenido.
class ProcessedChunk(TypedDict):
    page_content: str
    source: str

class RAGResult(TypedDict):
    question: str
    answer: str
    contexts: List[str] # Contextos solo con el texto (para Ragas)
    latency: float
    answer_tokens: int
    contexts_with_metadata: List[ProcessedChunk]
    unique_sources: List[str]
    
    # Añadimos claves que Ragas podría usar en su DataFrame, para facilitar el remapeo
    user_input: Optional[str] # Para la fusión de Ragas
    retrieved_contexts: Optional[List[str]] # Para la fusión de Ragas
    response: Optional[str] # Para la fusión de Ragas
    reference: Optional[str] # Para la fusión de Ragas

class RAGSystem(Protocol):
    """
    Protocolo para un sistema RAG evaluable.
    Cualquier RAG que queramos evaluar debe implementar este protocolo.
    """

    @property
    def llm_display_name(self) -> str:
        """Devuelve el nombre para mostrar del LLM utilizado por este RAG."""
        ...
        
    def query(self, question: str) -> RAGResult:
        """
        Realiza una consulta al sistema RAG.
        Debe devolver la respuesta, los contextos y métricas operacionales.
        """
        ... # La implementación real se hará en cada RAG específico


    