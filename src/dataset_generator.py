# /home/master/workspace/rag_sandbox_evaluation/src/dataset_generator.py
import json
import os
import pandas as pd
from typing import List, Dict, Tuple, Any
from datasets import Dataset


# from langchain_huggingface import HuggingFaceEmbeddings # Si necesitaras Embeddings aquí
# from langchain_chroma import Chroma # Si necesitaras Chroma aquí
# --------------------------------------------------------

# --- Importamos el LLM del juez (Gemma) para generar las preguntas ---
from src.components import load_llm # Para cargar el LLM
from src.config import LLM_MODEL_NAME, EMBED_MODEL_NAME # Para el LLM/Embeddings
import torch # Para torch_dtype en LLM

# Importamos el adaptador del RAG externo (para acceder a sus PDFs si se necesita)
import rag_sandbox_package as external_rag_package



def generate_questions_from_docs(
    documents: List[str],
    llm_instance_for_gen: Any, # El LLM que generará las preguntas (ej. Gemma)
    num_questions_per_doc: int = 1
) -> Tuple[List[str], List[str]]:
    questions = []
    ground_truths = []
    
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, # Podríamos reducirlo si el LLM tiene problemas de contexto
        chunk_overlap=50,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    print(f"🧠 Generando {num_questions_per_doc} preguntas por documento con el LLM generativo (usando: {llm_instance_for_gen.__class__.__name__})...")
    print(f"DEBUG_DATASET: Número de documentos pasados: {len(documents)}")

    # Iterar sobre una muestra de documentos para no sobrecargar el LLM en depuración    
    for i, doc_content in enumerate(documents[:5]): # Limitar a 5 documentos para probar
        doc_chunks = text_splitter.split_text(doc_content)
        print(f"DEBUG_DATASET: Documento {i+1} tiene {len(doc_chunks)} chunks.")
        
        for j, chunk in enumerate(doc_chunks[:num_questions_per_doc]):
            print(f"DEBUG_DATASET: Procesando chunk {j+1} del documento {i+1} (longitud: {len(chunk)}).")
            
            # Intentamos obligarle a ceñirse al formato.
            prompt = f"""
            Genera UNA pregunta y UNA respuesta basada EXCLUSIVAMENTE en el siguiente texto.
            La respuesta debe ser una verdad fundamental del texto.
            TU SALIDA DEBE SER EXCLUSIVAMENTE EN EL FORMATO ESPECIFICADO A CONTINUACIÓN.
            NO INCLUYAS NINGÚN OTRO TEXTO, COMENTARIOS NI EXPLICACIONES ANTES O DESPUÉS DEL FORMATO.

            <TEXTO_CONTEXTO>
            {chunk}
            </TEXTO_CONTEXTO>

            <FORMATO_SALIDA>
            Pregunta: [Tu pregunta aquí]
            Respuesta: [Tu respuesta aquí]
            </FORMATO_SALIDA>
            """
            # ----------------------------------------------------
            
            try:
                response = llm_instance_for_gen.invoke(prompt)
                full_response_text = response # Almacenamos la respuesta completa para debugging
                print(f"DEBUG_DATASET: RESPUESTA COMPLETA LLM para chunk {j+1}: ---INICIO---\n{full_response_text}\n---FIN---")
                
                # --- ¡AJUSTE DEL PARSER MÁS AGRESIVO! ---
                # Buscar el inicio de la sección de formato.
                format_start_tag = "<FORMATO_SALIDA>"
                format_end_tag = "</FORMATO_SALIDA>"

                if format_start_tag in full_response_text and format_end_tag in full_response_text:
                    format_section = full_response_text.split(format_start_tag)[1].split(format_end_tag)[0].strip()
                else:
                    # Si no encuentra las etiquetas, intentamos el parsing anterior (menos robusto)
                    format_section = full_response_text.strip()
                
                q_start = format_section.find("Pregunta:")
                a_start = format_section.find("Respuesta:")
                
                if q_start != -1 and a_start != -1 and a_start > q_start: # Asegurarse de que Respuesta viene después de Pregunta
                    # Extraer la pregunta desde el inicio de 'Pregunta:' hasta 'Respuesta:'
                    q_text = format_section[q_start + len("Pregunta:"):a_start].strip()
                    # Extraer la respuesta desde el inicio de 'Respuesta:' hasta el final
                    a_text = format_section[a_start + len("Respuesta:"):].strip()
                    
                    if q_text and a_text:
                        questions.append(q_text)
                        ground_truths.append(a_text)
                        print(f"   ✅ Generada P/R del documento {i+1}, chunk {j+1}: Q='{q_text[:50]}...' A='{a_text[:50]}...'")
                    else:
                        print(f"   ⚠️ Fallo al parsear P/R (vacío) del documento {i+1}, chunk {j+1}. LLM vació las partes. Respuesta LLM: {full_response_text[:100]}...")
                else:
                    print(f"   ⚠️ Fallo al encontrar 'Pregunta:' o 'Respuesta:' o formato incorrecto del documento {i+1}, chunk {j+1}. Respuesta LLM: {full_response_text[:100]}...")
            except Exception as e:
                print(f"   ❌ Error al generar P/R con LLM para chunk {j+1} del documento {i+1}: {e}")
                
    print(f"✅ Generadas {len(questions)} preguntas/respuestas en total.")
    return questions, ground_truths





def load_or_generate_dataset(
    dataset_path: str,
    rag_external_instance: Any,
    llm_for_question_gen: Any,
    force_regeneration: bool = True, # <--- ¡MANTENER EN TRUE TEMPORALMENTE PARA FORZAR LA REGENERACIÓN!
    num_questions_per_doc: int = 1
) -> Tuple[List[str], List[str]]:
    """
    Carga un dataset de preguntas/respuestas desde un archivo JSON,
    o lo genera a partir de los documentos del RAG externo si no existe o se fuerza la regeneración.
    """
    if os.path.exists(dataset_path) and not force_regeneration:
        print(f"📂 Cargando dataset de evaluación desde: {dataset_path}")
        try: # Añadimos un try/except para el json.load también
            with open(dataset_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            questions = data.get("questions", [])
            ground_truths = data.get("ground_truths", [])
            print(f"✅ Dataset cargado con {len(questions)} preguntas.")
            return questions, ground_truths
        except Exception as e:
            print(f"❌ Error al cargar el dataset JSON existente: {e}. Forzando regeneración.")
            # Si el JSON es inválido, lo tratamos como si no existiera y regeneramos.
            os.remove(dataset_path) # Eliminar archivo corrupto
            return load_or_generate_dataset(dataset_path, rag_external_instance, llm_for_question_gen, force_regeneration=True, num_questions_per_doc=num_questions_per_doc)
    else:
        # ... (código para obtener all_documents_content del RAG externo) ...
        print(f"🔄 Generando nuevo dataset de evaluación a partir de los documentos del RAG externo.")
        
        if rag_external_instance._rag_core_instance is None:
             raise RuntimeError("El RAG externo no está inicializado. No se pueden obtener documentos para generar preguntas.")
        
        try:
            chroma_docs_data = rag_external_instance._rag_core_instance.db.get(include=['documents', 'metadatas'])
            all_documents_content = [doc for doc in chroma_docs_data['documents']]
            
            if not all_documents_content:
                raise ValueError("No se encontraron documentos en la base de datos Chroma del RAG externo para generar preguntas.")

            questions, ground_truths = generate_questions_from_docs(
                documents=all_documents_content,
                llm_instance_for_gen=llm_for_question_gen,
                num_questions_per_doc=num_questions_per_doc
            )

            # --- Manejo explícito de dataset vacío después de generación ---
            if not questions:
                print("⚠️ ADVERTENCIA: generate_questions_from_docs no produjo ninguna pregunta.")
                # Devolvemos listas vacías, no None.
                return [], [] 
            # -----------------------------------------------------------

            with open(dataset_path, 'w', encoding='utf-8') as f:
                json.dump({"questions": questions, "ground_truths": ground_truths}, f, ensure_ascii=False, indent=4)
            print(f"✅ Dataset generado y guardado en: {dataset_path}")
            return questions, ground_truths
        
        except Exception as e:
            print(f"❌ ERROR al obtener documentos del RAG externo o generar el dataset: {e}. Devolviendo dataset vacío.")
            # Si hay un error al obtener/generar, devolver listas vacías para que el main no falle con None.
            return [], []