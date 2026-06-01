import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from src.agent_based_simulation.model import SindemiaViolenciaModel

st.set_page_config(layout="wide", page_title="Simulador: Arquitectura de la Exclusión")
st.title("Simulador de Interacción Institucional (ABS)")

if "df_prophet" not in st.session_state:
    st.warning(
        "No hay datos de pronóstico cargados en memoria. Por favor, ve a la página principal, ajusta los filtros y haz clic en 'Generar Pronóstico'."
    )
    st.stop()

df_prophet = st.session_state["df_prophet"]
sim_params = st.session_state["sim_params"]

st.markdown(
    f"**Contexto de la Simulación:** Municipio: {sim_params['municipio']} | Horizonte: {sim_params['dias']} días"
)

with st.sidebar:
    st.header("Intervención en Políticas Públicas")
    st.markdown(
        "Ajusta los recursos del sistema para observar el impacto en el subregistro."
    )
    capacidad = st.slider(
        "Capacidad Diaria (Atención x Inst.)", min_value=1, max_value=50, value=15
    )
    num_instituciones = st.slider(
        "Número de Puntos de Atención", min_value=1, max_value=20, value=5
    )

if st.button("Ejecutar Simulación", type="primary"):
    with st.spinner(
        "Procesando interacciones diarias entre entorno y sistema de salud..."
    ):

        modelo = SindemiaViolenciaModel(
            num_mujeres=1000,
            num_instituciones=num_instituciones,
            capacidad_institucional=capacidad,
            df_prophet=df_prophet,
        )

        for _ in range(sim_params["dias"]):
            modelo.step()

        resultados = modelo.datacollector.get_model_vars_dataframe()
        resultados["Fecha"] = df_prophet["Fecha"].values[: len(resultados)]

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=resultados["Fecha"],
                y=resultados["Violencia Real (Ocurrida)"],
                mode="lines",
                name="Realidad del Territorio (Oculta)",
                line=dict(color="#E53935", width=2),
            )
        )

        fig.add_trace(
            go.Scatter(
                x=resultados["Fecha"],
                y=resultados["Casos Capturados (SIVIGILA)"],
                mode="lines",
                name="Captura Efectiva Institucional",
                line=dict(color="#1E88E5", width=2),
            )
        )

        fig.add_trace(
            go.Scatter(
                x=pd.concat([resultados["Fecha"], resultados["Fecha"][::-1]]),
                y=pd.concat(
                    [
                        resultados["Violencia Real (Ocurrida)"],
                        resultados["Casos Capturados (SIVIGILA)"][::-1],
                    ]
                ),
                fill="toself",
                fillcolor="rgba(229, 57, 53, 0.2)",
                line=dict(color="rgba(255,255,255,0)"),
                name="Brecha (Subregistro Estructural)",
            )
        )

        fig.update_layout(
            title="Dinámica de Captura del Sistema frente a la Tendencia Real",
            xaxis_title="Fecha Proyectada",
            yaxis_title="Cantidad de Casos Diarios",
            template="plotly_white",
            hovermode="x unified",
        )

        st.plotly_chart(fig, use_container_width=True)

        st.info(
            "**Análisis de la Brecha:** El área roja sombreada ilustra de manera gráfica el volumen de víctimas que, por saturación de recursos institucionales o barreras geográficas asociadas a la ruralidad dispersa, no logran formalizar la notificación, generando los fenómenos de subregistro observados en la data histórica."
        )
