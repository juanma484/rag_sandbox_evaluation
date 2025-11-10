# /home/master/workspace/rag_sandbox_evaluation/src/config.py

import os

# --- Configuración General ---
PROJECT_NAME = "RAG Evaluation with Custom RAG"

OUTPUT_DIR = "output"
DASHBOARD_FILENAME = "rag_advanced_dashboard.html"

# --- Configuración del Dataset de Evaluación ---
# Ruta donde se espera encontrar un dataset JSON de preguntas y respuestas
# O donde se guardará si se genera automáticamente
EVAL_DATASET_PATH = os.path.join(OUTPUT_DIR, "evaluation_dataset.json")  

# --- Configuración LLM (para Ragas como Juez) ---
LLM_MODEL_NAME = "google/gemma-2-2b-it" # Gemma para el juez de Ragas
MAX_NEW_TOKENS = 200
TEMPERATURE = 0.7
TOP_K = 50
TOP_P = 0.95
REPETITION_PENALTY = 1.1

# --- Configuración LLM para Generación de Preguntas ---
# Usaremos Llama 3 8B Instruct para generar preguntas
LLM_GEN_MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"  

# --- Configuración Embeddings (para Ragas como Juez) ---
EMBED_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


os.makedirs(OUTPUT_DIR, exist_ok=True)