# main.py 
import os
import sys
import pandas as pd
from dotenv import load_dotenv

# Asegúrate de que sys.path esté bien configurado ANTES de importar tus módulos
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

# Importar desde tus módulos refactorizados
from src.config import OUTPUT_DIR, DASHBOARD_FILENAME, EVAL_DATASET_PATH, LLM_GEN_MODEL_NAME  
from src.components import load_llm, load_embeddings # load_llm cargará Llama 3
from src.rag_interface import RAGSystem, RAGResult
from src.my_custom_rag import ExternalRAGAdapter
from src.evaluation import run_ragas_evaluation
from src.dashboard_rag import crear_dashboard_evaluacion
from src.dataset_generator import load_or_generate_dataset

# Importar HuggingFacePipeline (ya que load_llm lo devuelve)
from langchain_huggingface import HuggingFacePipeline  

def main():
    load_dotenv(os.path.join(script_dir, '.env'))
    print("✅ Variables de entorno cargadas desde .env")

    hf_home_path = os.environ.get('HF_HOME') 

    if hf_home_path: 

        os.environ['HF_HOME'] = hf_home_path # Asegura que la variable de entorno se establezca 

        print(f"🏠 HF_HOME configurado a: {hf_home_path}") 

    else: 

        print("⚠️ HF_HOME no definido en .env. Usando el caché por defecto de Hugging Face.") 


    print(f"--- Iniciando {os.path.basename(os.getcwd())} ---")

    hf_token = os.environ.get('HF_TOKEN') # Este token es crucial para modelos como Llama 3

    if hf_token:
        print("🔑 Token de Hugging Face cargado de variables de entorno (primeros 5 caracteres):", hf_token[:5] + "*****")
    else:
        print("⚠️ HF_TOKEN no encontrado. Ragas y la generación de preguntas con Llama 3 podrían tener problemas.")
        raise ValueError("HF_TOKEN no configurado. Necesario para modelos Llama 3 y el juez de Ragas.") # <--- Es crítico

    # 1. Cargar Componentes para Ragas como Juez (LLM, Embeddings)
    # (Esto seguirá usando Gemma-2B-it)
    try:
        ragas_judge_llm, ragas_judge_tokenizer = load_llm(hf_token)
        ragas_judge_embeddings = load_embeddings(hf_token)
    except Exception as e:
        print(f"❌ Error crítico al cargar componentes para el juez de Ragas: {e}")
        return

    # --- Configurar LLM para Generación de Preguntas (¡Usando Llama 3!) ---
    llm_for_question_generation = None
    try:
        print(f"📥 Cargando LLM para generación de preguntas: {LLM_GEN_MODEL_NAME}...")
        # Reutilizamos load_llm que ya sabe cómo cargar modelos de Hugging Face
        # Podemos pasarle LLM_GEN_MODEL_NAME en lugar de LLM_MODEL_NAME
        llm_for_question_generation, _ = load_llm(hf_token, model_name_override=LLM_GEN_MODEL_NAME) # <--- Pasamos override
        print(f"✅ LLM '{LLM_GEN_MODEL_NAME}' configurado para generación de preguntas.")
    except Exception as e:
        print(f"❌ Error al cargar LLM para generación de preguntas ({LLM_GEN_MODEL_NAME}): {e}. La generación de preguntas fallará.")
        # Aquí, si falla, no podemos tener un fallback a Gemma, porque ya sabemos que no funciona.
        # Es mejor que falle o se detenga.
        raise # Es crítico que este LLM funcione para el dataset

    # 2. Inicializar tu RAG existente usando el adaptador
    print("\n--- Inicializando tu sistema RAG existente a través del adaptador ---")
    my_rag_adapter = ExternalRAGAdapter()
    external_rag_llm_name = my_rag_adapter.llm_display_name
    
    def rag_pipeline_wrapper(question: str) -> RAGResult:
        return my_rag_adapter.query(question)
    
    print(f"✅ Tu sistema RAG existente ({external_rag_llm_name}) listo para ser evaluado.")

    # --- 3. Cargar o Generar el Dataset de Evaluación ---
 
    eval_questions, ground_truths = load_or_generate_dataset(
        dataset_path=EVAL_DATASET_PATH,
        rag_external_instance=my_rag_adapter,
        llm_for_question_gen=llm_for_question_generation, # <--- ¡Pasamos el LLM Llama 3 a dataset_generator!
        force_regeneration=True, # Forzar regeneración la primera vez con Llama 3
        num_questions_per_doc=1
    )
    
    if not eval_questions:
        print("❌ ERROR CRÍTICO: No se pudo generar o cargar ningún par de pregunta/respuesta. La evaluación no puede continuar.")
        return
    # ----------------------------------------------------

    # 4. Ejecutar Evaluación Ragas
    final_results_df = run_ragas_evaluation(
        eval_questions=eval_questions, # <--- Usamos el dataset cargado/generado
        ground_truths=ground_truths, # <--- Usamos el dataset cargado/generado
        rag_pipeline_func=rag_pipeline_wrapper,
        llm_for_ragas=ragas_judge_llm,
        embeddings_for_ragas=ragas_judge_embeddings,
        model_name_for_dashboard=external_rag_llm_name
    )
    
    # 5. Generar Dashboard de Evaluación
    dashboard_output_path = os.path.join(OUTPUT_DIR, DASHBOARD_FILENAME)
    try:
        print(f"📊 Generando dashboard para RAG con LLM: {external_rag_llm_name}")
        crear_dashboard_evaluacion(
            final_results_df,
            output_path=dashboard_output_path,
            dashboard_title=f"Dashboard de Evaluación RAG - {external_rag_llm_name}",
            models_to_compare=[external_rag_llm_name]
        )
    except Exception as e:
        print(f"\n❌ Ocurrió un error al generar el dashboard: {e}")

    print("\n================================================================================")
    print("✅ Evaluación y Dashboard Completados.")
    print(f"📊 Dashboard interactivo guardado como: {os.path.abspath(dashboard_output_path)}")
    print("================================================================================")

if __name__ == "__main__":
    main()