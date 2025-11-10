# src/evaluation.py
import pandas as pd
import datetime
from typing import List, Dict, Callable
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
#from langchain_community.llms import HuggingFacePipeline
#from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_huggingface  import HuggingFacePipeline
from langchain_huggingface  import HuggingFaceEmbeddings


def run_ragas_evaluation(
    eval_questions: List[str],
    ground_truths: List[str],
    rag_pipeline_func: Callable[[str], Dict], # <--- Ahora solo acepta la pregunta y devuelve el Dict
    llm_for_ragas: HuggingFacePipeline, # <--- Nuevo nombre para el LLM del juez de Ragas
    embeddings_for_ragas: HuggingFaceEmbeddings, # <--- Nuevo nombre para los embeddings del juez de Ragas
    model_name_for_dashboard: str = "Modelo RAG Actual"
) -> pd.DataFrame:
    """
    Ejecuta la evaluación RAG utilizando Ragas y devuelve un DataFrame con los resultados.
    Ahora utiliza un rag_pipeline_func externo.
    """
    print("\n📊 Iniciando evaluación con Ragas...")

    eval_data = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": [],
        "latency": [],
        "answer_tokens": [],
        "simulated_satisfaction": [],
        "model_name": [],  
        "timestamp": [] 
        # "question_category": [], # Añadir si tienes categorías de preguntas
    }

    current_timestamp = datetime.datetime.now()

    print("🔄 Generando datos de evaluación con el RAG externo...")
    for i, q in enumerate(eval_questions):
        # Aquí llamas a tu función rag_pipeline que encapsula la lógica RAG
        result = rag_pipeline_func(q) # <--- Solo pasamos la pregunta aquí

        eval_data["question"].append(result["question"])
        eval_data["answer"].append(result["answer"])
        eval_data["contexts"].append(result["contexts"])
        eval_data["latency"].append(result["latency"])
        eval_data["answer_tokens"].append(result["answer_tokens"])
        eval_data["ground_truth"].append(ground_truths[i])
        eval_data["model_name"].append(model_name_for_dashboard) 
        eval_data["timestamp"].append(current_timestamp)  

        # Calcular la satisfacción simulada (1 si la respuesta contiene la verdad, 0 si no)
        satisfaction = 1 if ground_truths[i].lower() in result["answer"].lower() else 0
        eval_data["simulated_satisfaction"].append(satisfaction)

        # Si tuvieras categorías, las asignarías aquí
        # eval_data["question_category"].append("General")

    operational_df = pd.DataFrame(eval_data) # DataFrame con datos operativos y ground truths

    eval_dataset = Dataset.from_dict(eval_data)
    print(f"✅ Dataset de evaluación creado con {len(eval_dataset)} ejemplos.")

    print("🔧 Configurando Ragas para usar el LLM de Hugging Face y Embeddings locales (para juzgar)...")
    # Los wrappers de Ragas ahora usan los LLM y Embeddings pasados como argumentos
    ragas_llm = LangchainLLMWrapper(llm_for_ragas)
    ragas_embeddings = LangchainEmbeddingsWrapper(embeddings_for_ragas)

    metrics = [
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    ]

    print("🚀 Ejecutando evaluación de Ragas (Esto puede tardar unos minutos, el LLM está trabajando duro como juez)...")
    results = evaluate(
        dataset=eval_dataset,
        metrics=metrics,
        llm=ragas_llm,
        embeddings=ragas_embeddings,
        raise_exceptions=False # Importante para que no falle si el LLM no puede juzgar
    )

    print("\n================================================================================")
    print("📈 RESULTADOS DE LA EVALUACIÓN RAGAS (Local en Colab)")
    print("================================================================================\n")
    print("--- Métricas Promedio ---")
    print(results)

    ragas_metrics_df = results.to_pandas()

    if 'user_input' in ragas_metrics_df.columns:
        ragas_metrics_df.rename(columns={'user_input': 'question'}, inplace=True)
    if 'retrieved_contexts' in ragas_metrics_df.columns:
        ragas_metrics_df.rename(columns={'retrieved_contexts': 'contexts'}, inplace=True)
    if 'response' in ragas_metrics_df.columns:
        ragas_metrics_df.rename(columns={'response': 'answer'}, inplace=True)
    if 'reference' in ragas_metrics_df.columns:
        ragas_metrics_df.rename(columns={'reference': 'ground_truth'}, inplace=True)


    print("\n--- Columnas en DataFrame operacional (para la fusión) ---")
    print(operational_df.columns.tolist())
    print("\n--- Columnas en DataFrame de Ragas (después de to_pandas()) (para la fusión) ---")
    print(ragas_metrics_df.columns.tolist())

    if 'question' not in ragas_metrics_df.columns:
        print("\n⚠️  ADVERTENCIA: La columna 'question' no se encontró en los resultados de Ragas.")
        print("   Fusionando los DataFrames por índice, asumiendo que el orden de las filas es el mismo.")

        operational_df_indexed = operational_df.reset_index(drop=True)
        ragas_metrics_df_indexed = ragas_metrics_df.reset_index(drop=True)

        ragas_metric_cols_to_add = [
            col for col in ['faithfulness', 'answer_relevancy', 'context_precision', 'context_recall']
            if col in ragas_metrics_df_indexed.columns
        ]

        final_results_df = operational_df_indexed.copy()
        for col in ragas_metric_cols_to_add:
            final_results_df[col] = ragas_metrics_df_indexed[col]
    else:
        print("\n✅ La columna 'question' se encontró en los resultados de Ragas. Fusionando por 'question'.")
        # Asegurarse de no duplicar columnas si ya están presentes
        # Aquí la fusión es clave. Mantendremos las columnas de operational_df y añadiremos las de ragas_metrics_df
        # Evitando duplicar question, answer, contexts, ground_truth
        cols_to_keep_from_ragas = [col for col in ragas_metrics_df.columns if col not in operational_df.columns or col in ['question']]
        ragas_metrics_df_filtered = ragas_metrics_df[cols_to_keep_from_ragas]

        final_results_df = pd.merge(operational_df, ragas_metrics_df_filtered, on="question", how="left", suffixes=('_ops', '_ragas'))

        # Si hay columnas duplicadas como 'answer_ops' y 'answer_ragas', podemos consolidarlas
        for common_col in ['answer', 'contexts', 'ground_truth']:
            if f'{common_col}_ragas' in final_results_df.columns and f'{common_col}_ops' in final_results_df.columns:
                # Priorizar la versión original de operational_df si existe
                final_results_df[common_col] = final_results_df[f'{common_col}_ops'].fillna(final_results_df[f'{common_col}_ragas'])
                final_results_df.drop(columns=[f'{common_col}_ops', f'{common_col}_ragas'], inplace=True)
            elif f'{common_col}_ragas' in final_results_df.columns:
                final_results_df[common_col] = final_results_df[f'{common_col}_ragas']
                final_results_df.drop(columns=[f'{common_col}_ragas'], inplace=True)
            elif f'{common_col}_ops' in final_results_df.columns:
                final_results_df[common_col] = final_results_df[f'{common_col}_ops']
                final_results_df.drop(columns=[f'{common_col}_ops'], inplace=True)


    print("\n--- Detalles por Pregunta (DataFrame final) ---")
    for idx, row in final_results_df.iterrows():
        print(f"\n--- Ejemplo {idx+1}: {row['question']} ---")
        print(f"  Respuesta: {row['answer'][:60]}...")
        faithfulness_val = f"{row['faithfulness']:.4f}" if pd.notna(row['faithfulness']) else "N/A"
        answer_relevancy_val = f"{row['answer_relevancy']:.4f}" if pd.notna(row['answer_relevancy']) else "N/A"
        context_precision_val = f"{row['context_precision']:.4f}" if pd.notna(row['context_precision']) else "N/A"
        context_recall_val = f"{row['context_recall']:.4f}" if pd.notna(row['context_recall']) else "N/A"

        print(f"  Faithfulness: {faithfulness_val}")
        print(f"  Answer Relevancy: {answer_relevancy_val}")
        print(f"  Context Precision: {context_precision_val}")
        print(f"  Context Recall: {context_recall_val}")

    return final_results_df