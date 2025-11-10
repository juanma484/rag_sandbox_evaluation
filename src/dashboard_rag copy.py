# src/dashboard_rag.py
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import os

def crear_dashboard_evaluacion(results_df: pd.DataFrame, output_path: str = "output/rag_advanced_dashboard.html"):
    """
    Genera un dashboard HTML interactivo y avanzado para analizar los resultados de la evaluación RAG.
    """
    print(f"\n🚀 Creando dashboard avanzado de evaluación...")

    # --- 1. Preparación de Datos ---
    df_display = results_df.copy()
    metric_cols = ['faithfulness', 'answer_relevancy', 'context_precision', 'context_recall']
    operational_cols = ['latency', 'answer_tokens', 'simulated_satisfaction']
    all_metrics = metric_cols + operational_cols

    # --- 2. Creación de la Figura con un Layout Complejo ---
    fig = make_subplots(
        rows=5,
        cols=2,
        specs=[
            [{'type': 'indicator'}, {'type': 'indicator'}],
            [{'type': 'indicator'}, {'type': 'box', 'rowspan': 2}],
            [None, None], # Esto crea un espacio vacío si el rowspan solo cubre dos
            [{'type': 'heatmap'}, {'type': 'scatter'}],
            [{'type': 'table', 'colspan': 2}, None]
        ],
        row_heights=[0.15, 0.15, 0.15, 0.25, 0.3],
        subplot_titles=(
            "<b>Latencia Media (s)</b>", "<b>Tokens Medios por Respuesta</b>",
            "<b>Tasa de Satisfacción (Sim)</b>", "<b>Distribución de Puntuaciones Ragas</b>",
            None, # Subtítulo para la fila 3 que tiene None,None
            "<b>Correlación de Métricas</b>", "<b>Latencia vs. Tokens de Respuesta</b>",
            "<b>Resultados Detallados</b>"
        ),
        vertical_spacing=0.08, # Ajustado para mejor visualización
        horizontal_spacing=0.08
    )

    # --- 3. Fila de KPIs Operacionales ---
    fig.add_trace(go.Indicator(
        mode="number",
        value=df_display['latency'].mean(),
        number={'suffix': ' s', 'font': {'size': 40}},
        title={"text": "Latencia Media"}
    ), row=1, col=1)

    fig.add_trace(go.Indicator(
        mode="number",
        value=df_display['answer_tokens'].mean(),
        number={'font': {'size': 40}},
        title={"text": "Tokens Medios por Respuesta"}
    ), row=1, col=2)

    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=df_display['simulated_satisfaction'].mean() * 100,
        number={'suffix': ' %', 'font': {'size': 40}},
        title={"text": "Tasa de Satisfacción (Simulada)"},
        gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "#2ca02c"}}
    ), row=2, col=1)

    # --- 4. Gráficos de Análisis ---
    # Box Plots para distribuciones de Ragas
    # Solo añadir métricas si existen en el df y tienen valores no nulos
    ragas_present_metrics = [m for m in metric_cols if m in df_display.columns and df_display[m].notna().any()]
    if ragas_present_metrics:
        for metric in ragas_present_metrics:
            fig.add_trace(go.Box(
                y=df_display[metric],
                name=metric.replace('_', ' ').title(),
                boxpoints='all',
                jitter=0.3,
                pointpos=-1.8
            ), row=2, col=2)
        fig.update_yaxes(range=[0, 1.05], row=2, col=2)
    else:
        fig.add_annotation(
            text="No hay datos válidos de métricas Ragas para mostrar.",
            xref="x2", yref="y2", x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="red"), row=2, col=2
        )


    # Heatmap de Correlación
    # Asegúrate de que haya suficientes columnas numéricas y no todas sean NaN
    numeric_cols_for_corr = [col for col in all_metrics if col in df_display.columns and df_display[col].notna().any()]
    if len(numeric_cols_for_corr) > 1:
        corr = df_display[numeric_cols_for_corr].corr()
        fig.add_trace(go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.columns,
            colorscale=px.colors.diverging.RdBu,
            zmin=-1, zmax=1,
            text=corr.round(2).values,
            texttemplate="%{text}"
        ), row=4, col=1)
    else:
        fig.add_annotation(
            text="No hay suficientes datos numéricos para calcular la correlación.",
            xref="x4", yref="y4", x=0.5, y=0.5, showarrow=False,
            font=dict(size=14, color="gray"), row=4, col=1
        )


    # Scatter Plot de Latencia vs. Tokens
    if 'latency' in df_display.columns and 'answer_tokens' in df_display.columns and df_display['latency'].notna().any() and df_display['answer_tokens'].notna().any():
        fig.add_trace(go.Scatter(
            x=df_display['latency'],
            y=df_display['answer_tokens'],
            mode='markers',
            marker=dict(
                size=12,
                color=df_display['simulated_satisfaction'], # Colorear por satisfacción
                colorscale=[[0, 'red'], [1, 'green']],
                showscale=True,
                colorbar=dict(title='Satisfacción')
            ),
            text=df_display['question'].apply(lambda x: x[:50] + '...' if isinstance(x, str) else str(x)), # Mostrar pregunta en hover
            hoverinfo='text+x+y'
        ), row=4, col=2)
        fig.update_xaxes(title_text="Latencia (s)", row=4, col=2)
        fig.update_yaxes(title_text="Tokens en Respuesta", row=4, col=2)
    else:
        fig.add_annotation(
            text="No hay datos de latencia o tokens para mostrar.",
            xref="x5", yref="y5", x=0.5, y=0.5, showarrow=False,
            font=dict(size=14, color="gray"), row=4, col=2
        )

    # --- 5. Tabla de Resultados Detallados ---
    # Limitar el tamaño de los contextos para la tabla
    df_display['contexts'] = df_display['contexts'].apply(
        lambda x: '<br>'.join([f"- {c[:100]}..." for c in x]) if isinstance(x, list) else str(x)
    )
    for col in all_metrics:
        if col in df_display.columns:
            df_display[col] = df_display[col].round(3)
    
    table_cols = ['question', 'answer', 'ground_truth', 'contexts'] + all_metrics
    # Asegurarse de que las columnas existan antes de pasarlas a la tabla
    table_cols_present = [col for col in table_cols if col in df_display.columns]

    if not df_display.empty and table_cols_present:
        fig.add_trace(go.Table(
            header=dict(
                values=[f"<b>{col.replace('_', ' ').title()}</b>" for col in table_cols_present],
                fill_color='#1f77b4',
                align='left', font=dict(color='white', size=12)
            ),
            cells=dict(
                values=[df_display[col] for col in table_cols_present],
                fill_color='rgba(240, 240, 240, 0.95)',
                align='left', font=dict(color='black', size=11)
            )
        ), row=5, col=1)
    else:
        fig.add_annotation(
            text="No hay resultados detallados para mostrar en la tabla.",
            xref="x6", yref="y6", x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="red"), row=5, col=1
        )

    # --- 6. Diseño Final y Guardado ---
    fig.update_layout(
        title_text="📊 Dashboard Avanzado de Evaluación RAG",
        title_x=0.5,
        height=2200, # Aumentar altura un poco
        showlegend=False,
        margin=dict(l=40, r=40, t=100, b=40)
    )

    fig.write_html(output_path)
    abs_path = os.path.abspath(output_path)
    print(f"✅ Dashboard avanzado guardado con éxito en: {abs_path}")
    print("   Puedes encontrarlo en el panel de archivos a la izquierda y descargarlo.")