import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from src.agent_based_simulation.model import AuditoriaOperativaModel
from src.agent_based_simulation.agents import ExpedienteAgente

st.set_page_config(layout="wide", page_title="Auditoría Operativa SIVIGILA")
st.title("Simulador de Capacidad Institucional y Cuellos de Botella")

if "df_prophet" not in st.session_state or "sim_params" not in st.session_state:
    st.warning(
        "Genera el pronóstico en la página principal para cargar la matriz de datos."
    )
    st.stop()

df_prophet = st.session_state["df_prophet"]
sim_params = st.session_state["sim_params"]

st.markdown("""
**Auditoría Basada en Datos:** Modelo estricto de Teoría de Colas y Cadenas de Markov, alimentado por el pronóstico diario de Prophet y la matriz de remisiones reales extraída del SIVIGILA.
""")

with st.sidebar:
    st.header("1. Capacidad Base Diaria")
    cap_salud = st.slider("Cupos Salud Mental (Psicólogos)", 1, 50, 10)
    cap_prot = st.slider("Cupos Protección (Comisarías)", 1, 50, 15)

    st.header("2. Diseño de Turnos")
    pct_fin_semana = st.slider(
        "Retención Operativa Fines de Semana (%)",
        min_value=0.0,
        max_value=1.0,
        value=0.2,
        step=0.1,
        help="1.0 significa que trabajan al 100% sábados y domingos. 0.0 significa que las dependencias cierran por completo.",
    )

if st.button("Ejecutar Auditoría Operativa", type="primary"):
    with st.spinner("Procesando expedientes, triage empírico y calendario..."):

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
            pct_fin_semana=pct_fin_semana,
        )

        for _ in range(sim_params["dias"]):
            modelo.step()

        resultados = modelo.datacollector.get_model_vars_dataframe()
        resultados["Fecha"] = df_prophet["Fecha"].values[: len(resultados)]

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
            st.plotly_chart(fig1, width="stretch")

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
                title="Evolución del Backlog (Nota el efecto 'dientes de sierra' por los fines de semana)",
                template="plotly_white",
            )
            st.plotly_chart(fig2, width="stretch")

        st.markdown("---")
        st.markdown("### Auditoría de Inequidad y Triage Institucional")

        agentes_completados = [
            ag
            for ag in modelo.schedule.agents
            if isinstance(ag, ExpedienteAgente) and ag.esta_completado()
        ]
        fallos_administrativos = sum(
            [
                1
                for ag in modelo.schedule.agents
                if isinstance(ag, ExpedienteAgente) and ag.fallo_administrativo
            ]
        )

        if len(agentes_completados) > 0:
            df_hist = pd.DataFrame(
                {
                    "Dias Espera": [
                        ag.dias_en_espera_total for ag in agentes_completados
                    ],
                    "Grupo Prioritario": [
                        (
                            "Alta Prioridad (Menores/Sexual)"
                            if ag.prioridad == 1
                            else "Prioridad Regular"
                        )
                        for ag in agentes_completados
                    ],
                }
            )

            c1, c2 = st.columns([1, 2])

            with c1:
                st.metric(
                    "Total Demanda (Prophet)",
                    int(resultados["Nuevos Casos (Prophet)"].sum()),
                )
                st.metric(
                    "Expedientes Atrapados (Sin Resolver)",
                    int(resultados["Backlog Total"].iloc[-1]),
                )
                st.metric(
                    "Fallos Administrativos (> 30 días)",
                    fallos_administrativos,
                    delta_color="inverse",
                )

            with c2:
                fig_hist = px.histogram(
                    df_hist,
                    x="Dias Espera",
                    color="Grupo Prioritario",
                    marginal="box",
                    nbins=30,
                    opacity=0.7,
                    color_discrete_map={
                        "Alta Prioridad (Menores/Sexual)": "#D32F2F",
                        "Prioridad Regular": "#9E9E9E",
                    },
                    title="Distribución Real de Tiempos de Espera por Nivel de Vulnerabilidad",
                )
                fig_hist.update_layout(
                    template="plotly_white",
                    barmode="overlay",
                    xaxis_title="Días hasta resolución total",
                    yaxis_title="Cantidad de Víctimas",
                )
                st.plotly_chart(fig_hist, width="stretch")

            st.info(
                "**Análisis de Triage:** El modelo procesa primero a las víctimas de mayor vulnerabilidad (rojo). Sin embargo, si el sistema está muy colapsado, notarás que incluso la curva roja se desplaza hacia la derecha, demostrando que la falta de recursos vulnera hasta los casos más urgentes."
            )
