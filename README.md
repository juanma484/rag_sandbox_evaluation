# 📊 Framework de Evaluación de Sistemas RAG (Retrieval Augmented Generation)

Este proyecto es un framework extensible y automatizado diseñado para evaluar la calidad y el rendimiento de cualquier sistema RAG (Retrieval Augmented Generation). Proporciona herramientas para generar datasets de evaluación, ejecutar pruebas contra un RAG externo y visualizar los resultados a través de un dashboard interactivo.

## ✨ Características Principales

-   **Evaluación Flexible:** Diseñado para "enchufar y usar" cualquier sistema RAG externo que cumpla con una interfaz definida.
-   **Generación Automatizada de Datasets:** Utiliza Large Language Models (LLMs) avanzados (ej., Llama 3) para generar automáticamente preguntas y respuestas (ground truths) a partir de los documentos del RAG a evaluar.
-   **Métricas de Calidad RAG (Ragas):** Integra la librería Ragas para calcular métricas clave como:
    -   **Faithfulness (Fidelidad):** ¿La respuesta del RAG está respaldada por el contexto?
    -   **Answer Relevancy (Relevancia de la Respuesta):** ¿La respuesta es pertinente a la pregunta?
    -   **Context Precision (Precisión del Contexto):** ¿El contexto recuperado es relevante?
    -   **Context Recall (Exhaustividad del Contexto):** ¿Se recuperó todo el contexto necesario?
-   **Métricas Operacionales:** Incluye latencia y número de tokens generados.
-   **Dashboard Interactivo:** Genera un informe HTML completo con visualizaciones interactivas (medias, distribuciones, correlaciones) para un análisis profundo de los resultados.
-   **Integración de DeepEval (en el RAG Externo):** Fomenta la creación de pruebas unitarias y de regresión en el propio RAG externo para una verificación de calidad continua.

## 🚀 Cómo Empezar

### 1. Requisitos

-   Python 3.10+
-   `pip`
-   Acceso a una GPU (recomendado) para los LLMs open-source.
-   Un token válido de Hugging Face (requerido para modelos como Llama 3 y Gemma).
-   Acceso autorizado a los modelos restringidos de Hugging Face (ej., `meta-llama/Meta-Llama-3-8B-Instruct`, `google/gemma-2-2b-it`).
-   Opcional: Clave de API de OpenAI o Google si se prefiere para la generación de datasets o el juez de Ragas.

### 2. Estructura del Proyecto

rag_sandbox_evaluation/
├── main.py # Punto de entrada principal para la evaluación
├── requirements.txt # Dependencias del framework de evaluación
├── .env # Variables de entorno (HF_TOKEN, ELASTIC_API_KEY, etc.)
├── src/
│ ├── init.py
│ ├── components.py # Carga de LLMs y Embeddings (juez Ragas)
│ ├── rag_interface.py # Definición de la interfaz RAGSystem y RAGResult
│ ├── my_custom_rag.py # Adaptador para el RAG externo a evaluar
│ ├── evaluation.py # Lógica de ejecución de Ragas
│ ├── dashboard_rag.py # Generación del dashboard HTML
│ └── dataset_generator.py # Generación/carga del dataset de evaluación
├── output/ # Salidas del dashboard y dataset
└── .venv/ # Entorno virtual de Python

### 3. Configuración

#### a. Configurar tu RAG Externo

Asegúrate de que tu sistema RAG externo (ubicado en `/home/master/workspace/rag_sandbox/`):
-   Está estructurado como un paquete Python (ej. `rag_system_core`).
-   Tiene un `setup.py` que lista todas sus dependencias.
-   Define un punto de entrada (ej. `rag_system_core/rag_entrypoint.py`) que expone `initialize_application()`, `rag_core_instance`, `llm_display_name`, `active_pdf_filters` a nivel de paquete (vía `__init__.py`).
-   Su `rag_core.py` devuelve un diccionario `RAGResult` desde `process_query`.
-   Su `.env` contiene las configuraciones específicas (rutas de PDFs, ChromaDB/Elasticsearch, LLM) para que funcione de forma independiente.

#### b. Configurar Variables de Entorno (`.env`)

En la raíz de este proyecto (`rag_sandbox_evaluation/.env`), crea un archivo `.env` con tus credenciales:

HF_TOKEN="hf_YOUR_HUGGINGFACE_TOKEN_HERE"

Opcional, si usas OpenAI para generar el dataset

OPENAI_API_KEY="sk-YOUR_OPENAI_API_KEY_HERE"

Opcional, si has movido tus modelos de HF a una carpeta específica

HF_HOME="/path/to/your/huggingface_models/"

#### c. Configurar `src/config.py`

Revisa `src/config.py` para ajustar los nombres de los LLMs (para el juez Ragas y la generación de dataset) y otras rutas de salida.

### 4. Instalación

1.  **Navega a la raíz de este proyecto:**
    ```bash
    cd /home/master/workspace/rag_sandbox_evaluation
    ```
2.  **Crea y activa un entorno virtual:**
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    pip install --upgrade pip setuptools wheel
    ```
3.  **Instala las dependencias, incluyendo el RAG externo como paquete editable:**
    ```bash
    pip install -r requirements.txt
    ```
    (Asegúrate de que tu `requirements.txt` contiene la línea `-e /home/master/workspace/rag_sandbox` que apunta a la raíz de tu RAG externo).

### 5. Ejecución

1.  **Limpia los cachés (esencial después de cambios):**
    ```bash
    rm -rf __pycache__ src/__pycache__
    rm -rf /home/master/workspace/rag_sandbox/__pycache__ /home/master/workspace/rag_sandbox/rag_system_core/__pycache__ /home/master/workspace/rag_sandbox/rag_system_core.egg-info
    ```
2.  **Activa el entorno virtual** (si no lo está).
3.  **Ejecuta el script principal de evaluación:**
    ```bash
    python main.py
    ```

## 📈 Resultados

El dashboard interactivo `rag_advanced_dashboard.html` se generará en la carpeta `output/` de este proyecto. Ábrelo en tu navegador web para visualizar las métricas y los análisis.
