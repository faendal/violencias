import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from src.agent_based_simulation.model import SindemiaViolenciaModel

st.set_page_config(layout="wide", page_title="Simulador: Arquitectura de la Exclusión")
st.title("Simulador Sindicémico y Cuellos de Botella")

if "df_prophet" not in st.session_state or "sim_params" not in st.session_state:
    st.warning(
        "Debes generar primero un pronóstico en la página principal para cargar el sembrado empírico."
    )
    st.stop()

df_prophet = st.session_state["df_prophet"]
sim_params = st.session_state["sim_params"]

st.markdown(
    f"**Sembrado Empírico Activo:** Municipio: `{sim_params['municipio']}` | Simulación de `{sim_params['dias']}` días"
)

with st.sidebar:
    st.header("Ruta Institucional: Sector Salud")
    num_inst_salud = st.number_input("Puntos de Atención (Salud)", 1, 20, 2)
    cap_salud = st.slider("Capacidad Diaria (Salud)", 1, 50, 10, key="csalud")

    st.header("Ruta Institucional: Sector Justicia")
    num_inst_justicia = st.number_input("Comisarías / Fiscalía", 1, 20, 5)
    cap_justicia = st.slider("Capacidad Diaria (Justicia)", 1, 50, 15, key="cjusticia")

if st.button("Ejecutar Simulación Dinámica", type="primary"):
    with st.spinner("Procesando agentes y ciclos de retroalimentación..."):

        modelo = SindemiaViolenciaModel(
            num_mujeres=2000,
            num_inst_salud=num_inst_salud,
            num_inst_justicia=num_inst_justicia,
            cap_salud=cap_salud,
            cap_justicia=cap_justicia,
            df_prophet=df_prophet,
            sim_params=sim_params,
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
                    y=resultados["Violencia Real (Ocurrida)"],
                    mode="lines",
                    name="Realidad Oculta",
                    line=dict(color="#E53935"),
                )
            )
            fig1.add_trace(
                go.Scatter(
                    x=resultados["Fecha"],
                    y=resultados["Capturas SIVIGILA"],
                    mode="lines",
                    name="Registro SIVIGILA",
                    line=dict(color="#1E88E5"),
                )
            )
            fig1.add_trace(
                go.Scatter(
                    x=pd.concat([resultados["Fecha"], resultados["Fecha"][::-1]]),
                    y=pd.concat(
                        [
                            resultados["Violencia Real (Ocurrida)"],
                            resultados["Capturas SIVIGILA"][::-1],
                        ]
                    ),
                    fill="toself",
                    fillcolor="rgba(229, 57, 53, 0.1)",
                    line=dict(color="rgba(255,255,255,0)"),
                    name="Brecha",
                )
            )
            fig1.update_layout(
                title="Subregistro Estructural Global", template="plotly_white"
            )
            st.plotly_chart(fig1, use_container_width=True)

        with col2:
            fig2 = go.Figure()
            fig2.add_trace(
                go.Scatter(
                    x=resultados["Fecha"],
                    y=resultados["Brecha Saturación"],
                    mode="lines",
                    stackgroup="one",
                    name="Rechazo por Saturación",
                    fillcolor="#FFB74D",
                )
            )
            fig2.add_trace(
                go.Scatter(
                    x=resultados["Fecha"],
                    y=resultados["Abandono Geográfico"],
                    mode="lines",
                    stackgroup="one",
                    name="Abandono Geográfico (Rural)",
                    fillcolor="#81C784",
                )
            )
            fig2.add_trace(
                go.Scatter(
                    x=resultados["Fecha"],
                    y=resultados["Abandono Desconfianza"],
                    mode="lines",
                    stackgroup="one",
                    name="Abandono por Desconfianza",
                    fillcolor="#E57373",
                )
            )
            fig2.update_layout(
                title="Anatomía de la Exclusión (Causas de no denuncia)",
                template="plotly_white",
            )
            st.plotly_chart(fig2, width="stretch")

        col3, col4 = st.columns(2)

        with col3:
            fig3 = px.area(
                resultados,
                x="Fecha",
                y="Revictimización",
                title="Casos Crónicos (Víctimas de múltiples ataques)",
                color_discrete_sequence=["#8E24AA"],
            )
            fig3.update_layout(template="plotly_white")
            st.plotly_chart(fig3, use_container_width=True)

        with col4:
            st.markdown("### Métricas Finales de Impacto")
            st.metric(
                "Total Agresiones Simuladas",
                int(resultados["Violencia Real (Ocurrida)"].sum()),
            )
            st.metric(
                "Capturadas por SIVIGILA", int(resultados["Capturas SIVIGILA"].sum())
            )
            eficiencia = (
                (
                    resultados["Capturas SIVIGILA"].sum()
                    / resultados["Violencia Real (Ocurrida)"].sum()
                )
                * 100
                if resultados["Violencia Real (Ocurrida)"].sum() > 0
                else 0
            )
            st.metric("Tasa de Efectividad del Sistema", f"{eficiencia:.1f}%")
            st.info(
                "**Conclusión:** Un porcentaje bajo de efectividad demuestra la necesidad urgente de fortalecer las rutas diferenciadas, ya que el backlog institucional genera un ciclo de desconfianza que alimenta la revictimización crónica observada a la izquierda."
            )
