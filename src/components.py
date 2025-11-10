# src/components.py

import os
import logging
from typing import Tuple, Dict, Optional, Any, Union, Type

from langchain_huggingface import HuggingFacePipeline
from langchain_huggingface import HuggingFaceEmbeddings

from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import torch

from .config import LLM_MODEL_NAME, EMBED_MODEL_NAME, \
    MAX_NEW_TOKENS, TEMPERATURE, TOP_K, TOP_P, REPETITION_PENALTY


# Añadido un argumento opcional model_name_override
def load_llm(hf_token: str | None, model_name_override: Optional[str] = None) -> Tuple[HuggingFacePipeline, AutoTokenizer]: # <--- ¡NUEVO PARÁMETRO!
    """
    Carga el Large Language Model (LLM) de Hugging Face y su tokenizer.
    Permite sobrescribir el nombre del modelo.
    """
    model_to_load = model_name_override if model_name_override else LLM_MODEL_NAME
    print(f"📥 Cargando modelo LLM de Hugging Face: {model_to_load} (Esto puede demorar la primera vez)...")
    try:
        if not hf_token and "gemma" in model_to_load.lower() or "llama" in model_to_load.lower(): # <--- Añadido Llama
            print(f"⚠️ Advertencia: El modelo '{model_to_load}' es gated (restringido) y no se proporcionó HF_TOKEN.")
            print("   Esto podría fallar. Asegúrate de haber aceptado los términos en Hugging Face y de que tu token es válido.")
            # Para Llama 3, el token es estrictamente necesario, así que lanzamos un error si no está
            if "llama" in model_to_load.lower() and not hf_token:
                raise ValueError(f"HF_TOKEN es requerido para cargar el modelo '{model_to_load}'.")

        tokenizer = AutoTokenizer.from_pretrained(model_to_load, token=hf_token)

        model = AutoModelForCausalLM.from_pretrained(
            model_to_load,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            token=hf_token
            # load_in_8bit=True # Descomentar para cuantificación 8-bit y ahorrar RAM
            # load_in_4bit=True # Descomentar para cuantificación 4-bit (requiere más ajustes)
        )

        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=TEMPERATURE,
            top_k=TOP_K,
            top_p=TOP_P,
            repetition_penalty=REPETITION_PENALTY,
            pad_token_id=tokenizer.eos_token_id,
        )

        local_llm = HuggingFacePipeline(pipeline=pipe)
        print(f"✅ Modelo LLM '{model_to_load}' de Hugging Face cargado y listo.")
        return local_llm, tokenizer
    except Exception as e:
        print(f"❌ ERROR al cargar o inicializar el modelo LLM de Hugging Face '{model_to_load}': {e}")
        print("Sugerencia: Asegúrate de tener `T4 GPU` en tu entorno de ejecución de Colab (o un entorno con GPU configurado).")
        print("Si el problema persiste, intenta un modelo más pequeño o investiga la cuantificación (load_in_8bit/4bit).")
        print("También, verifica que tu `HF_TOKEN` sea válido si estás usando un modelo privado o alcanzando límites de tasa.")
        raise

def load_embeddings(hf_token: str | None) -> HuggingFaceEmbeddings:
    """
    Carga el modelo de embeddings de Hugging Face.
    Recibe el token de Hugging Face como argumento.
    """
    print(f"📥 Cargando modelo de embeddings: {EMBED_MODEL_NAME} (Esto puede demorar la primera vez)...")
    try:
        local_embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME, model_kwargs={'token': hf_token})
        _ = local_embeddings.embed_query("test")
        print(f"✅ Embeddings cargados. Dimensión: {len(local_embeddings.embed_query('test'))}")
        return local_embeddings
    except Exception as e:
        print(f"❌ ERROR al cargar o inicializar el modelo de embeddings: {e}")
        raise

