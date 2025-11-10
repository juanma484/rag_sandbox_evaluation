# --- START OF FILE demo_dashboard.py ---
import pandas as pd
import os
import sys
from datetime import datetime, timedelta
import random

# --- Configuración para Colab y rutas ---
# Ajusta esta ruta a la raíz de tu proyecto en Drive
project_root = '/home/master/workspace/rag_sandbox_evaluation'

# Añadir la raíz del proyecto y src al sys.path para importaciones
if project_root not in sys.path:
    sys.path.insert(0, project_root)
src_path = os.path.join(project_root, 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

print(f"sys.path actualizado: {sys.path[:3]}...")

# Importa la función del dashboard desde tu nuevo módulo
try:
    from src.demo_dashboard_rag import crear_dashboard_evaluacion
    print("✅ Función 'crear_dashboard_evaluacion' importada con éxito desde 'demo_dashboard_rag'.")
except ImportError as e:
    print(f"❌ Error al importar 'crear_dashboard_evaluacion': {e}")
    print("Asegúrate de que 'src/demo_dashboard_rag.py' existe y '__init__.py' está en 'src/'.")
    sys.exit(1) # Salir si no podemos importar la función

# --- 1. Generar Datos de Prueba (Simulando results_df con más dimensiones) ---
print("\n⚙️ Generando datos de prueba para la demo del dashboard con datos temporales y categorías...")

# Definimos un conjunto de preguntas comunes y sus categorías
questions_info = [
    {"question": "¿Quién creó Python?", "category": "Tecnología"},
    {"question": "¿Qué necesita una planta para la fotosíntesis?", "category": "Ciencia"},
    {"question": "¿Qué es un Sprint en Agile?", "category": "Gestión de Proyectos"},
    {"question": "¿Qué es un LLM?", "category": "Tecnología"},
    {"question": "¿Cuál es la capital de Francia?", "category": "Geografía"},
    {"question": "¿Quién escribió 'Cien años de soledad'?", "category": "Literatura"},
    {"question": "¿Qué base de datos vectorial es escalable?", "category": "Tecnología"},
    {"question": "¿Es Python un lenguaje compilado?", "category": "Tecnología"},
]

# Definimos las fechas de evaluación
evaluation_dates = [
    datetime(2025, 8, 1),
    datetime(2025, 9, 1),
    datetime(2025, 9, 20) # Añadir una tercera fecha para ver más evolución
]

all_demo_data = []

# Ground truths y contextos de referencia (simplificados para demo)
# En un escenario real, esto se recuperaría de algún lugar
reference_data = {
    "¿Quién creó Python?": {
        "ground_truth": "Python fue creado por Guido van Rossum en 1991.",
        "contexts": ["Python es un lenguaje de programación interpretado, de alto nivel y de propósito general. Creado por Guido van Rossum y lanzado por primera vez en 1991."]
    },
    "¿Qué necesita una planta para la fotosíntesis?": {
        "ground_truth": "Las plantas usan luz solar, agua y dióxido de carbono para la fotosíntesis.",
        "contexts": ["La fotosíntesis es el proceso mediante el cual las plantas usan la luz solar, el agua y el dióxido de carbono para crear su propio alimento."]
    },
    "¿Qué es un Sprint en Agile?": {
        "ground_truth": "Un Sprint es una iteración corta en el desarrollo ágil de software.",
        "contexts": ["El desarrollo ágil de software se basa en iteraciones cortas llamadas Sprints. Scrum es un marco popular para implementar Agile."]
    },
    "¿Qué es un LLM?": {
        "ground_truth": "Un Large Language Model (LLM) es un modelo de lenguaje con muchos parámetros.",
        "contexts": ["Un Large Language Model (LLM) es un modelo de lenguaje con muchos parámetros, capaz de entender y generar texto similar al humano."]
    },
    "¿Cuál es la capital de Francia?": {
        "ground_truth": "La capital de Francia es París.",
        "contexts": ["La capital de Francia es París."]
    },
    "¿Quién escribió 'Cien años de soledad'?": {
        "ground_truth": "Gabriel García Márquez fue el autor de 'Cien años de soledad'.",
        "contexts": ["'Cien años de soledad' es una novela del escritor colombiano Gabriel García Márquez."]
    },
    "¿Qué base de datos vectorial es escalable?": {
        "ground_truth": "Milvus es una base de datos vectorial de código abierto altamente escalable.",
        "contexts": ["Milvus es una base de datos vectorial de código abierto altamente escalable, diseñada para gestionar embeddings de aprendizaje automático."]
    },
    "¿Es Python un lenguaje compilado?": {
        "ground_truth": "Python es un lenguaje interpretado.",
        "contexts": ["Python es un lenguaje de programación interpretado, de alto nivel y de propósito general."]
    },
}


# Generar datos para cada modelo, pregunta y fecha
for eval_date in evaluation_dates:
    for model_name in ['Modelo_A', 'Modelo_B', 'Modelo_C_Experimental']: # Añadir un tercer modelo para más complejidad
        for q_info in questions_info:
            question = q_info['question']
            category = q_info['category']
            ref_data = reference_data.get(question, {"ground_truth": "N/A", "contexts": ["N/A"]})

            # Simular respuestas y métricas con variaciones plausibles
            # y que mejoren/empeoren ligeramente con el tiempo o entre modelos
            base_faithfulness = 0.85
            base_relevancy = 0.85
            base_precision = 0.85
            base_recall = 0.85
            base_latency = 1.0
            base_tokens = 60
            base_satisfaction = 1 # assume good initially

            # Introduce variaciones basadas en el modelo y la fecha
            if model_name == 'Modelo_A':
                # Modelo A es bastante estable
                faith = base_faithfulness + random.uniform(-0.05, 0.05)
                relev = base_relevancy + random.uniform(-0.05, 0.05)
                prec = base_precision + random.uniform(-0.05, 0.05)
                rec = base_recall + random.uniform(-0.05, 0.05)
                lat = base_latency + random.uniform(-0.2, 0.2)
                tok = base_tokens + random.randint(-10, 10)
            elif model_name == 'Modelo_B':
                # Modelo B mejora con el tiempo, pero es más lento
                # Mejoras simuladas para fechas posteriores
                date_factor = (eval_date - evaluation_dates[0]).days / 30 # ~0, 1, 2
                faith = (base_faithfulness - 0.1) + date_factor * 0.05 + random.uniform(-0.03, 0.03)
                relev = (base_relevancy - 0.05) + date_factor * 0.03 + random.uniform(-0.03, 0.03)
                prec = (base_precision - 0.05) + date_factor * 0.02 + random.uniform(-0.03, 0.03)
                rec = (base_recall - 0.05) + date_factor * 0.02 + random.uniform(-0.03, 0.03)
                lat = (base_latency + 0.3) + date_factor * -0.05 + random.uniform(-0.1, 0.1) # Más lento, pero mejora un poco
                tok = (base_tokens + 15) + random.randint(-5, 5) # Más tokens
            else: # Modelo_C_Experimental - más volátil, puede ser muy bueno o muy malo
                date_factor = (eval_date - evaluation_dates[0]).days / 30
                faith = (base_faithfulness + random.uniform(-0.2, 0.1)) + date_factor * 0.01
                relev = (base_relevancy + random.uniform(-0.2, 0.1)) + date_factor * 0.01
                prec = (base_precision + random.uniform(-0.2, 0.1)) + date_factor * 0.01
                rec = (base_recall + random.uniform(-0.2, 0.1)) + date_factor * 0.01
                lat = (base_latency + random.uniform(-0.5, 0.5)) + date_factor * -0.01
                tok = (base_tokens + random.randint(-20, 20))

            # Asegurar que las métricas de 0 a 1 estén en ese rango
            faith = max(0.0, min(1.0, faith))
            relev = max(0.0, min(1.0, relev))
            prec = max(0.0, min(1.0, prec))
            rec = max(0.0, min(1.0, rec))

            # Simular la satisfacción basada en faithfulness y relevancy
            satis = 1 if (faith > 0.75 and relev > 0.75) else 0

            # Simular respuestas variadas
            simulated_answer = f"Respuesta simulada de {model_name} para '{question}' en {eval_date.strftime('%Y-%m-%d')}."


            all_demo_data.append({
                'timestamp': eval_date,
                'model_name': model_name,
                'question_category': category,
                'question': question,
                'answer': simulated_answer,
                'ground_truth': ref_data['ground_truth'],
                'contexts': ref_data['contexts'],
                'faithfulness': faith,
                'answer_relevancy': relev,
                'context_precision': prec,
                'context_recall': rec,
                'latency': lat,
                'answer_tokens': tok,
                'simulated_satisfaction': satis
            })

demo_results_df = pd.DataFrame(all_demo_data)

# Convertir timestamp a formato de fecha para visualización si es necesario
demo_results_df['timestamp'] = pd.to_datetime(demo_results_df['timestamp'])


# --- 2. Preparar Directorio de Salida ---
output_dir = os.path.join(project_root, 'output')
os.makedirs(output_dir, exist_ok=True)
output_filename = os.path.join(output_dir, "rag_advanced_dashboard_demo.html")

# --- 3. Generar el Dashboard ---
print(f"🚀 Generando dashboard de demostración en: {output_filename}")

metric_descriptions = {
    'faithfulness': "Mide la precisión y la consistencia de la respuesta del modelo **RAG** con respecto a la información proporcionada en los documentos fuente. Una puntuación alta indica que la respuesta no contiene alucinaciones y se basa directamente en el contexto recuperado.",
    'answer_relevancy': "Evalúa qué tan directa y completamente responde la respuesta generada por el **modelo RAG** a la pregunta del usuario. Ignora la veracidad de la respuesta.",
    'context_precision': "Indica qué tan relevante es el contexto recuperado para responder a la pregunta. Una puntuación alta significa que los pasajes recuperados son directamente útiles.",
    'context_recall': "Mide la exhaustividad del contexto recuperado, es decir, si todos los hechos necesarios para responder la pregunta están presentes en el contexto.",
    'latency': "Tiempo en segundos que tarda el modelo **RAG** en generar una respuesta.",
    'answer_tokens': "Número de tokens en la respuesta generada por el **modelo RAG**.",
    'simulated_satisfaction': "Métrica binaria (0/1) que indica la satisfacción simulada del usuario con la respuesta del **modelo RAG**. Puede representar una evaluación humana simplificada."
}

crear_dashboard_evaluacion(
    df=demo_results_df,
    output_path=output_filename,
    dashboard_title="Evaluación Comparativa de Sistemas RAG (Temporal y por Categoría)", # Título más explícito
    metric_descriptions=metric_descriptions,
    models_to_compare=demo_results_df['model_name'].unique().tolist()
)

print("\n✅ Demo del dashboard generada con éxito.")
print(f"Abre el archivo '{output_filename}' en tu navegador web para verlo.")
# --- END OF FILE demo_dashboard.py ---