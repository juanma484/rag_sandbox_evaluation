# --- START OF FILE src/demo_dashboard_rag.py ---
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import pandas as pd
import numpy as np
import os

def crear_dashboard_evaluacion(
    df: pd.DataFrame,
    output_path: str,
    dashboard_title: str = "Dashboard de Evaluación RAG",
    metric_descriptions: dict = None,
    models_to_compare: list = None
):
    """
    Genera un dashboard interactivo HTML para la evaluación de modelos RAG.

    Args:
        df (pd.DataFrame): DataFrame con los resultados de la evaluación,
                          incluyendo 'question', 'answer', 'ground_truth', 'contexts',
                          métricas de Ragas (faithfulness, answer_relevancy, etc.)
                          y 'model_name' para comparación.
        output_path (str): Ruta completa donde se guardará el archivo HTML del dashboard.
        dashboard_title (str): Título principal del dashboard.
        metric_descriptions (dict): Diccionario con descripciones para cada métrica.
        models_to_compare (list): Lista de nombres de modelos para comparar.
    """
    if metric_descriptions is None:
        metric_descriptions = {}

    if models_to_compare is None or not models_to_compare:
        models_to_compare = df['model_name'].unique().tolist()
        if not models_to_compare:
            models_to_compare = ["Modelo Único"]
            if 'model_name' not in df.columns:
                df['model_name'] = models_to_compare[0]

    # Asegurarse de que la columna 'timestamp' sea de tipo datetime para gráficos temporales
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])


    # --- Configuración General de Estilos ---
    plotly_template = go.layout.Template()
    plotly_template.layout.font.size = 12
    plotly_template.layout.title.font.size = 22
    plotly_template.layout.hoverlabel.font.size = 10

    # ¡NUEVA PALETA DE COLORES CUALITATIVOS EN GAMAS TIERRA!
    # Colores para diferenciar los modelos
    earth_tone_qualitative_colors = [
        '#8B4513', # Saddle Brown
        '#A0522D', # Sienna
        '#6B8E23', # Olive Drab
        '#CD853F', # Peru
        '#D2B48C', # Tan
        '#BC8F8F', # Rosy Brown
        '#BDB76B', # Dark Khaki
        '#8FBC8F'  # Dark Sea Green
    ]
    model_colors = {model: earth_tone_qualitative_colors[i % len(earth_tone_qualitative_colors)] for i, model in enumerate(models_to_compare)}


    # Calcular promedios para identificar el mejor modelo (ej. por faithfulness)
    avg_faithfulness_per_model = df.groupby('model_name')['faithfulness'].mean()
    best_model_faithfulness = ""
    if not avg_faithfulness_per_model.empty:
        best_model_name = avg_faithfulness_per_model.idxmax()
        best_model_score = avg_faithfulness_per_model.max()
        best_model_faithfulness = f"Actualmente, <strong>{best_model_name}</strong> muestra la mayor fidelidad promedio ({best_model_score:.2f})."


    # --- Encabezado del Dashboard ---
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>{dashboard_title}</title>
        <link href="https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&display=swap" rel="stylesheet">
        <style>
            body {{ font-family: 'Roboto', sans-serif; margin: 20px; background-color: #fcfbf8; color: #36454F; }} /* Fondo crema, texto gris-marrón */
            h1 {{ font-size: 32pt; text-align: center; color: #5c4033; margin-bottom: 30px; border-bottom: 2px solid #a17a6a; padding-bottom: 15px; }} /* Marrón oscuro para título, línea marrón */
            h2 {{ font-size: 24pt; color: #7e6357; margin-top: 40px; border-bottom: 1px solid #dcdcdc; padding-bottom: 10px; }} /* Marrón medio para subtítulos */
            h3 {{ font-size: 18pt; color: #7e6357; margin-top: 30px; }} /* Marrón medio para títulos de sección */
            .section {{ background-color: #f5f2ed; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,.1); padding: 20px; margin-bottom: 25px; }} /* Fondo de sección más claro */
            .kpi-container {{ display: flex; justify-content: space-around; flex-wrap: wrap; margin-bottom: 20px; }}
            .kpi-box {{ background-color: #e8e2d7; border-radius: 5px; padding: 15px 20px; margin: 10px; text-align: center; flex: 1; min-width: 200px; max-width: 28%; box-shadow: inset 0 0 5px rgba(0,0,0,.05); }} /* Fondo KPI caja tono arena */
            .kpi-label {{ font-size: 14pt; color: #696969; margin-bottom: 5px; }} /* Gris medio para etiquetas KPI */
            .kpi-value {{ font-size: 24pt; font-weight: 700; color: #a0522d; }} /* Sienna para valores KPI */
            .metric-description {{ font-size: 12pt; line-height: 1.6; color: #696969; margin-bottom: 10px; }}
            .interactive-tip {{ background-color: #e0efe0; border-left: 5px solid #6b8e23; padding: 15px; margin: 20px 0; font-size: 13pt; color: #36454F; }} /* Verde oliva sutil para el tip */
            table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
            th, td {{ border: 1px solid #d3d3d3; padding: 8px; text-align: left; vertical-align: top; }} /* Bordes de tabla más suaves */
            th {{ background-color: #e8e2d7; font-weight: bold; color: #5c4033; }} /* Cabeceras de tabla tono arena, texto marrón */
            .dataframe-table {{ font-size: 11pt; }}
            details {{ margin-top: 20px; background-color: #f5f2ed; border: 1px solid #d3d3d3; border-radius: 5px; padding: 10px; }} /* Detalles con fondo de sección */
            summary {{ font-weight: bold; cursor: pointer; color: #7e6357; font-size: 16pt; }} /* Título detalles marrón medio */
            details p {{ margin: 5px 0 5px 20px; }}
            /* Estilo para layout compacto de gráficos */
            .charts-grid {{ display: flex; flex-wrap: wrap; justify-content: space-around; gap: 20px; }}
            .charts-grid > div {{ flex: 1 1 45%; min-width: 400px; }}

            /* Nuevos estilos para layout compacto de resúmenes de modelo */
            .model-summaries-grid {{ display: flex; flex-wrap: wrap; justify-content: space-around; gap: 20px; margin-top: 20px; }}
            .model-summary-card {{
                flex: 1 1 30%;
                min-width: 300px;
                background-color: #edeae3; /* Fondo tarjeta modelo ligeramente más claro */
                border-radius: 8px;
                padding: 15px;
                box-shadow: 0 1px 3px rgba(0,0,0,.08);
                text-align: center;
            }}
            .model-summary-card h3 {{ margin-top: 0; margin-bottom: 15px; color: #5c4033; font-size: 20pt; }} /* Marrón oscuro para título tarjeta */
            .model-summary-card .kpi-container {{ flex-wrap: wrap; justify-content: center; gap: 5px; margin-bottom: 0; }}
            .model-summary-card .kpi-box {{
                margin: 5px;
                padding: 10px 15px;
                flex: 1 1 auto;
                min-width: 120px;
                max-width: 45%;
                font-size: 10pt;
            }}
            .model-summary-card .kpi-label {{ font-size: 10pt; color: #696969; }}
            .model-summary-card .kpi-value {{ font-size: 18pt; color: #a0522d; }}
        </style>
    </head>
    <body>
        <h1>{dashboard_title}</h1>
        <p style="text-align: center; font-size: 16pt; color: #555; margin-bottom: 30px;">
            Este dashboard ofrece una visión integral de la evaluación comparativa de diferentes
            <strong>sistemas RAG (Retrieval Augmented Generation)</strong>. Nuestro objetivo es ayudarte a
            entender el rendimiento de cada modelo en métricas clave de calidad y eficiencia,
            facilitando la identificación de fortalezas, debilidades y oportunidades de mejora.
            Los "Modelos" comparados representan distintas configuraciones o versiones de tu pipeline RAG.
            {best_model_faithfulness}
            <br>
            Este informe incluye análisis de evolución temporal y segmentación por categoría de pregunta para un diagnóstico más profundo.
        </p>
        <div class="interactive-tip">
            <strong>Consejo:</strong> Las gráficas son interactivas. Pase el ratón sobre los elementos para ver detalles, haga clic en la leyenda para ocultar/mostrar series, y use las herramientas de zoom/pan en la parte superior derecha de cada gráfica para explorar los datos.
        </div>
    """

    # --- 2. Glosario de Métricas ---
    if metric_descriptions:
        html_content += """
        <div class="section">
            <details>
                <summary>Glosario de Métricas de Evaluación RAG</summary>
                <div style="padding: 10px 0;">
                    <p class="metric-description">
                        Aquí se detallan las métricas clave utilizadas para evaluar el rendimiento de los modelos RAG.
                        Comprender cada métrica es fundamental para interpretar correctamente los resultados y tomar decisiones informadas.
                    </p>
        """
        for metric, desc in metric_descriptions.items():
            html_content += f"<p><strong>{metric.replace('_', ' ').title()}:</strong> {desc}</p>"
        html_content += """
                </div>
            </details>
        </div>
        """

    # --- 3. KPIs Resumen por Modelo ---
    html_content += """
    <h2>Rendimiento Global por Modelo</h2>
    <p class="metric-description">
        Esta sección presenta un resumen de las métricas promedio para cada modelo RAG evaluado.
        Estos indicadores clave de rendimiento (KPIs) permiten una visión rápida de alto nivel
        sobre el desempeño general de cada implementación y facilitan la comparación inicial.
    </p>
    <div class="section">
        <div class="model-summaries-grid">
    """
    for model in models_to_compare:
        df_model = df[df['model_name'] == model]
        if not df_model.empty:
            avg_metrics = df_model[['faithfulness', 'answer_relevancy', 'context_precision', 'context_recall', 'latency']].mean()
            html_content += f"""
            <div class="model-summary-card">
                <h3>{model}</h3>
                <div class="kpi-container">
                    <div class="kpi-box"><div class="kpi-label">Faithfulness</div><div class="kpi-value">{avg_metrics['faithfulness']:.2f}</div></div>
                    <div class="kpi-box"><div class="kpi-label">Answer Relevancy</div><div class="kpi-value">{avg_metrics['answer_relevancy']:.2f}</div></div>
                    <div class="kpi-box"><div class="kpi-label">Context Precision</div><div class="kpi-value">{avg_metrics['context_precision']:.2f}</div></div>
                    <div class="kpi-box"><div class="kpi-label">Context Recall</div><div class="kpi-value">{avg_metrics['context_recall']:.2f}</div></div>
                    <div class="kpi-box"><div class="kpi-label">Latencia (s)</div><div class="kpi-value">{avg_metrics['latency']:.2f}</div></div>
                </div>
            </div>
            """
        else:
            html_content += f"""
            <div class="model-summary-card">
                <h3>{model}</h3>
                <p class="metric-description">No hay datos disponibles.</p>
            </div>
            """
    html_content += "</div></div>"


    # --- 4. Gráficas Comparativas de Métricas RAG ---
    html_content += """
    <h2>Comparación de Métricas RAG entre Modelos</h2>
    <div class="section">
        <p class="metric-description">
            Este gráfico de barras agrupa las puntuaciones promedio de las métricas de calidad RAG
            (Fidelidad, Relevancia de Respuesta, Precisión y Exhaustividad de Contexto)
            para cada modelo. Permite una comparación visual directa entre los modelos
            para identificar sus fortalezas relativas en los aspectos clave de la generación RAG.
        </p>
    """
    avg_df = df.groupby('model_name')[['faithfulness', 'answer_relevancy', 'context_precision', 'context_recall']].mean().reset_index()
    avg_df_melted = avg_df.melt(id_vars='model_name', var_name='Métrica', value_name='Puntuación Promedio')

    fig_metrics = px.bar(
        avg_df_melted,
        x='Métrica',
        y='Puntuación Promedio',
        color='model_name',
        barmode='group',
        title='Puntuación Promedio de Métricas RAG por Modelo',
        template=plotly_template,
        color_discrete_map=model_colors # Usa la nueva paleta
    )
    fig_metrics.update_layout(
        title_font_size=20,
        xaxis_title_font_size=16,
        yaxis_title_font_size=16,
        legend_title_text='Modelo',
        yaxis_range=[0,1]
    )
    html_content += fig_metrics.to_html(full_html=False, include_plotlyjs='cdn')
    html_content += "</div>"

    # --- 5. Gráficas de Métricas Operacionales ---
    html_content += """
    <h2>Análisis de Métricas Operacionales</h2>
    <div class="section">
        <p class="metric-description">
            Esta sección visualiza las métricas relacionadas con la eficiencia operativa de los modelos RAG,
            como la latencia (tiempo de respuesta) y el número de tokens generados en promedio.
            El gráfico de dispersión ayuda a explorar la relación entre la latencia y la fidelidad
            de la respuesta para cada pregunta, lo cual es crucial para entender los compromisos
            entre rendimiento y calidad.
        </p>
        <div class="charts-grid">
    """
    avg_ops_df = df.groupby('model_name')[['latency', 'answer_tokens']].mean().reset_index()

    fig_latency = px.bar(
        avg_ops_df,
        x='model_name',
        y='latency',
        color='model_name',
        title='Latencia Promedio por Modelo (s)',
        template=plotly_template,
        color_discrete_map=model_colors # Usa la nueva paleta
    )
    fig_latency.update_layout(title_font_size=20, xaxis_title_text='Modelo', yaxis_title_text='Latencia (s)')
    html_content += f"<div>{fig_latency.to_html(full_html=False, include_plotlyjs='cdn')}</div>"

    fig_tokens = px.bar(
        avg_ops_df,
        x='model_name',
        y='answer_tokens',
        color='model_name',
        title='Promedio de Tokens por Respuesta',
        template=plotly_template,
        color_discrete_map=model_colors # Usa la nueva paleta
    )
    fig_tokens.update_layout(title_font_size=20, xaxis_title_text='Modelo', yaxis_title_text='Tokens')
    html_content += f"<div>{fig_tokens.to_html(full_html=False, include_plotlyjs='cdn')}</div>"

    fig_scatter = px.scatter(
        df,
        x="latency",
        y="faithfulness",
        color="model_name",
        hover_name="question",
        hover_data=['answer', 'ground_truth', 'contexts', 'faithfulness', 'answer_relevancy', 'context_precision', 'context_recall'],
        title="Latencia vs. Fidelidad (Faithfulness) por Pregunta y Modelo",
        labels={"latency": "Latencia (s)", "faithfulness": "Faithfulness"},
        template=plotly_template,
        color_discrete_map=model_colors # Usa la nueva paleta
    )
    fig_scatter.update_layout(title_font_size=20)
    html_content += f"<div style='flex: 1 1 95%;'>{fig_scatter.to_html(full_html=False, include_plotlyjs='cdn')}</div>"
    html_content += "</div></div>"

    # --- Nueva Sección: Evolución Temporal del Rendimiento ---
    if 'timestamp' in df.columns and len(df['timestamp'].unique()) > 1:
        html_content += """
        <h2>Evolución Temporal del Rendimiento</h2>
        <div class="section">
            <p class="metric-description">
                Esta sección permite observar cómo las métricas clave de los modelos RAG han evolucionado a lo largo del tiempo.
                Es fundamental para monitorear mejoras, detectar regresiones o entender el impacto de las actualizaciones en el sistema.
            </p>
            <div class="charts-grid">
        """
        temporal_avg_df = df.groupby(['timestamp', 'model_name'])[['faithfulness', 'answer_relevancy', 'latency', 'simulated_satisfaction']].mean().reset_index()
        temporal_avg_df = temporal_avg_df.sort_values(by=['timestamp', 'model_name'])

        fig_time_faith = px.line(
            temporal_avg_df,
            x='timestamp',
            y='faithfulness',
            color='model_name',
            title='Evolución de Faithfulness Promedio',
            template=plotly_template,
            color_discrete_map=model_colors, # Usa la nueva paleta
            markers=True
        )
        fig_time_faith.update_layout(title_font_size=20, xaxis_title_text='Fecha de Evaluación', yaxis_title_text='Faithfulness Promedio')
        html_content += f"<div>{fig_time_faith.to_html(full_html=False, include_plotlyjs='cdn')}</div>"

        fig_time_latency = px.line(
            temporal_avg_df,
            x='timestamp',
            y='latency',
            color='model_name',
            title='Evolución de Latencia Promedio (s)',
            template=plotly_template,
            color_discrete_map=model_colors, # Usa la nueva paleta
            markers=True
        )
        fig_time_latency.update_layout(title_font_size=20, xaxis_title_text='Fecha de Evaluación', yaxis_title_text='Latencia Promedio (s)')
        html_content += f"<div>{fig_time_latency.to_html(full_html=False, include_plotlyjs='cdn')}</div>"

        if 'simulated_satisfaction' in temporal_avg_df.columns:
            fig_time_satisfaction = px.line(
                temporal_avg_df,
                x='timestamp',
                y='simulated_satisfaction',
                color='model_name',
                title='Evolución de Satisfacción Promedio',
                template=plotly_template,
                color_discrete_map=model_colors, # Usa la nueva paleta
                markers=True
            )
            fig_time_satisfaction.update_layout(title_font_size=20, xaxis_title_text='Fecha de Evaluación', yaxis_title_text='Satisfacción Promedio')
            html_content += f"<div>{fig_time_satisfaction.to_html(full_html=False, include_plotlyjs='cdn')}</div>"

        html_content += "</div></div>"

    # --- Nueva Sección: Rendimiento por Categoría de Pregunta ---
    if 'question_category' in df.columns and len(df['question_category'].unique()) > 1:
        html_content += """
        <h2>Rendimiento por Categoría de Pregunta</h2>
        <div class="section">
            <p class="metric-description">
                Este análisis desglosa el rendimiento de los modelos RAG por diferentes categorías de preguntas.
                Es crucial para identificar en qué tipos de preguntas cada modelo sobresale o, por el contrario,
                presenta debilidades, permitiendo optimizaciones más dirigidas a casos de uso específicos.
            </p>
            <div class="charts-grid">
        """
        category_avg_df = df.groupby(['question_category', 'model_name'])[['faithfulness', 'answer_relevancy', 'latency']].mean().reset_index()

        fig_cat_faith = px.bar(
            category_avg_df,
            x='question_category',
            y='faithfulness',
            color='model_name',
            barmode='group',
            title='Faithfulness Promedio por Categoría',
            template=plotly_template,
            color_discrete_map=model_colors # Usa la nueva paleta
        )
        fig_cat_faith.update_layout(title_font_size=20, xaxis_title_text='Categoría de Pregunta', yaxis_title_text='Faithfulness Promedio', yaxis_range=[0,1])
        html_content += f"<div>{fig_cat_faith.to_html(full_html=False, include_plotlyjs='cdn')}</div>"

        fig_cat_latency = px.bar(
            category_avg_df,
            x='question_category',
            y='latency',
            color='model_name',
            barmode='group',
            title='Latencia Promedio por Categoría (s)',
            template=plotly_template,
            color_discrete_map=model_colors # Usa la nueva paleta
        )
        fig_cat_latency.update_layout(title_font_size=20, xaxis_title_text='Categoría de Pregunta', yaxis_title_text='Latencia Promedio (s)')
        html_content += f"<div>{fig_cat_latency.to_html(full_html=False, include_plotlyjs='cdn')}</div>"

        html_content += "</div></div>"


    # --- Análisis de Correlación de Métricas ---
    html_content += """
    <h2>Análisis de Correlación entre Métricas</h2>
    <div class="section">
        <p class="metric-description">
            Esta matriz de calor muestra la correlación de Pearson entre las diferentes métricas de evaluación.
            Los valores cercanos a 1 o -1 indican una fuerte correlación positiva o negativa, respectivamente,
            sugiriendo que las métricas tienden a aumentar/disminuir juntas o en direcciones opuestas.
            Los valores cercanos a 0 indican poca o ninguna relación lineal.
            Esto es útil para entender cómo la mejora en una métrica puede impactar en otras o identificar posibles compromisos
            (por ejemplo, si aumentar la latencia mejora significativamente la fidelidad, o viceversa).
        </p>
    """
    correlation_cols = ['faithfulness', 'answer_relevancy', 'context_precision', 'context_recall',
                        'latency', 'answer_tokens', 'simulated_satisfaction']

    existing_numeric_cols = [col for col in correlation_cols if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]

    if len(existing_numeric_cols) > 1:
        corr_matrix = df[existing_numeric_cols].corr(numeric_only=True)

        fig_corr = px.imshow(
            corr_matrix,
            text_auto=True,
            aspect="auto",
            color_continuous_scale=px.colors.sequential.RdBu, # Mantener RdBu para la correlación es bueno semánticamente
            title='Matriz de Correlación de Métricas',
            template=plotly_template
        )
        fig_corr.update_layout(
            title_font_size=20,
            xaxis_tickangle=-45,
            xaxis_title="", yaxis_title=""
        )
        fig_corr.update_xaxes(side="top")
        html_content += fig_corr.to_html(full_html=False, include_plotlyjs='cdn')
    else:
        html_content += "<p class='metric-description'>No hay suficientes métricas numéricas para calcular la matriz de correlación.</p>"
    html_content += "</div>"


    # --- 6. Tabla Detallada de Resultados Individuales ---
    html_content += """
    <h2>Análisis Detallado de Preguntas Individuales</h2>
    <div class="section dataframe-table">
        <p class="metric-description">
            Esta tabla presenta los resultados completos de la evaluación para cada pregunta, modelo y fecha de evaluación.
            Las entradas están ordenadas por pregunta, modelo y fecha para facilitar la comparación directa.
            Utiliza esta sección para analizar casos específicos, identificar fallos o éxitos,
            y profundizar en las respuestas, contextos recuperados y puntuaciones detalladas.
            Puedes observar cómo se comportan los diferentes modelos ante la misma pregunta en distintos momentos.
        </p>
    """
    display_df = df[['timestamp', 'model_name', 'question_category', 'question', 'answer', 'ground_truth', 'contexts',
                     'faithfulness', 'answer_relevancy', 'context_precision', 'context_recall',
                     'latency', 'answer_tokens', 'simulated_satisfaction']].copy()

    display_df['timestamp'] = display_df['timestamp'].dt.strftime('%Y-%m-%d')

    display_df = display_df.sort_values(by=['question', 'timestamp', 'model_name']).reset_index(drop=True)

    for col in ['faithfulness', 'answer_relevancy', 'context_precision', 'context_recall']:
        display_df[col] = display_df[col].apply(lambda x: f"{x:.2f}")
    for col in ['latency']:
        display_df[col] = display_df[col].apply(lambda x: f"{x:.2f}s")
    for col in ['answer_tokens']:
        display_df[col] = display_df[col].apply(lambda x: f"{int(x)}")
    display_df['simulated_satisfaction'] = display_df['simulated_satisfaction'].apply(lambda x: '✅ Satisfactorio' if x == 1 else '❌ Insatisfactorio')

    display_df['contexts'] = display_df['contexts'].apply(lambda x: "<br>".join(x) if isinstance(x, list) else (x if pd.notna(x) else ''))

    html_content += display_df.to_html(
        index=False,
        escape=False,
        classes='dataframe-table',
        table_id='detailed_results_table'
    )
    html_content += """
        <script>
            // Simple JS para hacer la tabla searchable/sortable si se desea integrar librerías como DataTables.js
            // Por ahora, solo es una tabla HTML estática mejorada con CSS.
        </script>
    </div>
    """

    # --- Pie de página ---
    html_content += """
    </body>
    </html>
    """

    # --- Guardar el archivo ---
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print(f"Dashboard guardado en: {output_path}")

# --- END OF FILE src/demo_dashboard_rag.py ---