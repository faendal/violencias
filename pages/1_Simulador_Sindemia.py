import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from src.agent_based_simulation.model import AuditoriaOperativaModel

st.set_page_config(layout="wide", page_title="Auditoría Operativa SIVIGILA")
st.title("Simulador de Capacidad Institucional (Enrutamiento Empírico)")

if "df_prophet" not in st.session_state or "sim_params" not in st.session_state:
    st.warning(
        "⚠️ Genera el pronóstico en la página principal para cargar la matriz de datos."
    )
    st.stop()

df_prophet = st.session_state["df_prophet"]
sim_params = st.session_state["sim_params"]

st.markdown("""
**Arquitectura del Modelo:** Esta simulación extrae la cantidad diaria de casos de **Prophet** y determina si requieren atención psicológica o de comisarías utilizando una matriz de transición de Markov basada en el cruce de las variables `naturaleza`, `ac_mental` y `remit_prot` de la base original del SIVIGILA.
""")

with st.sidebar:
    st.header("Capacidad Diaria del Sistema")
    st.markdown("Ajusta el número de cupos diarios disponibles para todo el municipio.")
    cap_salud = st.slider("Cupos Salud Mental (Psicólogos)", 1, 50, 10)
    cap_prot = st.slider("Cupos Protección (Comisarías)", 1, 50, 15)

if st.button("Ejecutar Auditoría Operativa", type="primary"):
    with st.spinner("Procesando expedientes y simulando colas institucionales..."):

        # Matriz de Markov extraída de los datos crudos (Simplificada para el ejemplo)
        # En la realidad, las agresiones físicas demandan mucha protección, las negligencias mucha salud mental.
        matriz_markov_empirica = {
            "Violencia Física": {"Salud Mental": 0.25, "Proteccion": 0.57},
            "Violencia Psicológica": {"Salud Mental": 0.88, "Proteccion": 0.81},
            "Negligencia": {"Salud Mental": 0.90, "Proteccion": 0.77},
            "Violencia Sexual": {"Salud Mental": 0.85, "Proteccion": 0.90},
        }

        capacidades_sistema = {"Salud Mental": cap_salud, "Proteccion": cap_prot}

        modelo = AuditoriaOperativaModel(
            df_prophet=df_prophet,
            sim_params=sim_params,
            matriz_markov=matriz_markov_empirica,
            capacidades=capacidades_sistema,
        )

        for _ in range(sim_params["dias"]):
            modelo.step()

        resultados = modelo.datacollector.get_model_vars_dataframe()
        resultados["Fecha"] = df_prophet["Fecha"].values[: len(resultados)]

        # --- TABLERO DE RESULTADOS ---
        col1, col2 = st.columns(2)

        with col1:
            fig1 = go.Figure()
            fig1.add_trace(
                go.Scatter(
                    x=resultados["Fecha"],
                    y=resultados["Nuevos Casos (Prophet)"].cumsum(),
                    mode="lines",
                    name="Demanda Acumulada",
                    line=dict(color="black", dash="dash"),
                )
            )
            fig1.add_trace(
                go.Scatter(
                    x=resultados["Fecha"],
                    y=resultados["Casos Completados"],
                    mode="lines",
                    name="Expedientes Resueltos",
                    fill="tozeroy",
                    fillcolor="rgba(46, 125, 50, 0.3)",
                    line=dict(color="#2E7D32"),
                )
            )
            fig1.update_layout(
                title="Rendimiento Global del Sistema", template="plotly_white"
            )
            st.plotly_chart(fig1, use_container_width=True)

        with col2:
            fig2 = go.Figure()
            fig2.add_trace(
                go.Scatter(
                    x=resultados["Fecha"],
                    y=resultados["Backlog Salud Mental"],
                    mode="lines",
                    stackgroup="one",
                    name="Represa: Salud Mental",
                    fillcolor="#1976D2",
                )
            )
            fig2.add_trace(
                go.Scatter(
                    x=resultados["Fecha"],
                    y=resultados["Backlog Proteccion"],
                    mode="lines",
                    stackgroup="one",
                    name="Represa: Comisarías (Protección)",
                    fillcolor="#F57C00",
                )
            )
            fig2.update_layout(
                title="Descomposición del Cuello de Botella Operativo",
                template="plotly_white",
            )
            st.plotly_chart(fig2, use_container_width=True)

        # Gráfico Inferior y Métricas
        c1, c2, c3 = st.columns(3)
        c1.metric(
            "Total Demanda (Prophet)", int(resultados["Nuevos Casos (Prophet)"].sum())
        )
        c2.metric(
            "Casos Atrapados en el Sistema", int(resultados["Backlog Total"].iloc[-1])
        )

        fallos = int(resultados["Fallo Administrativo (>30 días)"].iloc[-1])
        c3.metric(
            "Fallo Administrativo (> 30 días en cola)", fallos, delta_color="inverse"
        )

        if fallos > 0:
            st.error(
                f"🚨 **Alerta de Colapso:** Las capacidades actuales generaron que {fallos} expedientes superaran los 30 días legales de espera, entrando en riesgo de impunidad administrativa o revictimización."
            )
