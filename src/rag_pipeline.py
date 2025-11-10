# src/rag_pipeline.py
import time
from typing import List, Dict, Tuple
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain_community.llms import HuggingFacePipeline
from langchain_community.embeddings import HuggingFaceEmbeddings
from pymilvus import MilvusClient
from transformers import AutoTokenizer

from src.config import CHUNK_SIZE, CHUNK_OVERLAP, RETRIEVAL_K, COLLECTION_NAME

prompt_template = PromptTemplate.from_template(
    """
Responde la pregunta basándote *únicamente* en el siguiente contexto. Si el contexto no contiene la respuesta, di "No lo sé".

Contexto:
{context}

Pregunta: {question}

Respuesta:
"""
)

def process_documents(documents: List[str]) -> List[Dict]:
    """
    Divide los documentos en chunks y los prepara para la indexación.
    """
    print("✂️ Dividiendo documentos en chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = []
    for i, doc_text in enumerate(documents):
        doc_chunks = text_splitter.split_text(doc_text)
        for j, chunk_text in enumerate(doc_chunks):
            chunks.append({
                "id": f"doc{i}_chunk{j}",
                "text": chunk_text
            })
    print(f"✅ Creados {len(chunks)} chunks.")
    return chunks

def index_documents(
    milvus_client: MilvusClient,
    local_embeddings: HuggingFaceEmbeddings,
    chunks: List[Dict]
):
    """
    Genera embeddings para los chunks e los indexa en Milvus.
    """
    print(f"📤 Generando embeddings e indexando en Milvus (Colección: {COLLECTION_NAME})...")
    texts_to_embed = [chunk["text"] for chunk in chunks]
    all_embeddings = local_embeddings.embed_documents(texts_to_embed)

    data_to_insert = []
    for i, chunk in enumerate(chunks):
        data_to_insert.append({
            "id": i, # Usamos el índice como ID para Milvus
            "vector": all_embeddings[i],
            "text": chunk["text"],
            "source_id": chunk["id"] # ID original del chunk para trazabilidad
        })

    milvus_client.insert(collection_name=COLLECTION_NAME, data=data_to_insert)
    print(f"✅ Indexación completada. Milvus tiene ahora {milvus_client.get_collection_stats(COLLECTION_NAME)['row_count']} chunks.")

def retrieve_context(
    milvus_client: MilvusClient,
    local_embeddings: HuggingFaceEmbeddings,
    question: str,
    k: int = RETRIEVAL_K
) -> List[str]:
    """
    Recupera contextos relevantes de Milvus para una pregunta.
    """
    # print(f"\n🔍 Recuperando contexto para: '{question}'")
    query_embedding = local_embeddings.embed_query(question)
    search_results = milvus_client.search(
        collection_name=COLLECTION_NAME,
        data=[query_embedding],
        limit=k,
        output_fields=["text"]
    )
    retrieved_texts = [res["entity"]["text"] for res in search_results[0]]
    return retrieved_texts

def generate_answer(
    local_llm: HuggingFacePipeline,
    tokenizer: AutoTokenizer,
    question: str,
    context_list: List[str]
) -> Tuple[str, int]:
    """
    Genera una respuesta utilizando el LLM y el contexto recuperado.
    """
    context_str = "\n\n".join(context_list)
    prompt = prompt_template.format(context=context_str, question=question)

    # print("🤖 Generando respuesta con el LLM de Hugging Face...")
    response = local_llm.invoke(prompt)

    # Calcular el número de tokens en la respuesta como proxy de coste
    answer_tokens = len(tokenizer.encode(response))

    return response.strip(), answer_tokens

def rag_pipeline(
    question: str,
    milvus_client: MilvusClient,
    local_embeddings: HuggingFaceEmbeddings,
    local_llm: HuggingFacePipeline,
    tokenizer: AutoTokenizer
) -> Dict:
    """
    Ejecuta el pipeline RAG completo para una pregunta.
    """
    start_time = time.time() # Iniciar cronómetro

    contexts = retrieve_context(milvus_client, local_embeddings, question)
    answer, answer_tokens = generate_answer(local_llm, tokenizer, question, contexts)

    end_time = time.time() # Detener cronómetro
    latency = end_time - start_time

    return {
        "question": question,
        "contexts": contexts,
        "answer": answer,
        "latency": latency,
        "answer_tokens": answer_tokens
    }