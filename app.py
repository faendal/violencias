from pathlib import Path
from typing import Tuple

import pandas as pd
import streamlit as st
from prophet import Prophet
import plotly.graph_objects as go

# Ruta al consolidado mensual
DATA_PATH = Path("data/processed/SerieMensual.csv")


# -------------------------------------------------------------------
# Carga de datos
# -------------------------------------------------------------------
@st.cache_data
def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Carga la serie mensual consolidada con todos los campos necesarios."""
    df = pd.read_csv(path, parse_dates=["fecha"])

    # Aseguramos tipos básicos
    df["anio_hecho"] = df["anio_hecho"].astype(int)
    df["mes_num"] = df["mes_num"].astype(int)
    df["nombre_municipio"] = df["nombre_municipio"].astype(str)
    df["estrato"] = df["estrato"].astype(str)
    df["sexo_victima"] = df["sexo_victima"].astype(str)
    df["sexo_agresor"] = df["sexo_agresor"].astype(str)

    return df


# -------------------------------------------------------------------
# Entrenamiento del modelo (similar al legacy)
# -------------------------------------------------------------------
@st.cache_resource
def train_model_for(
    municipio: str,
    estrato: str,
    sexo_victima: str,
    sexo_agresor: str,
) -> Tuple[Prophet, pd.DataFrame]:
    """
    Filtra según los cuatro niveles (o 'Todos'), agrupa por mes,
    entrena un Prophet y devuelve (modelo, ts) donde ts tiene ['ds','y'] mensual.
    """
    df = load_data()
    df_f = df.copy()

    if municipio != "Todos":
        df_f = df_f[df_f["nombre_municipio"] == municipio]
    if estrato != "Todos":
        df_f = df_f[df_f["estrato"] == estrato]
    if sexo_victima != "Todos":
        df_f = df_f[df_f["sexo_victima"] == sexo_victima]
    if sexo_agresor != "Todos":
        df_f = df_f[df_f["sexo_agresor"] == sexo_agresor]

    if df_f.empty:
        raise ValueError("No hay datos para la combinación de filtros seleccionada.")

    # Agregamos por fecha (mensual), por si los filtros no dejan una sola fila por mes
    ts = (
        df_f.groupby("fecha", as_index=False)["casos"]
        .sum()
        .rename(columns={"fecha": "ds", "casos": "y"})
        .sort_values("ds")
    )

    if ts["y"].sum() == 0:
        raise ValueError("La serie resultante tiene 0 casos en todo el histórico.")

    # Modelo Prophet con estacionalidad anual (mensual)
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
    )
    model.fit(ts)

    return model, ts


# -------------------------------------------------------------------
# Función principal
# -------------------------------------------------------------------
def main() -> None:
    st.set_page_config(
        layout="wide",
        page_title="Pronóstico de Casos de Violencia",
    )

    st.title("Pronóstico De Casos De Violencia En El Departamento De Antioquia")

    df = load_data()

    st.sidebar.header("Filtros")

    # 1) Municipio
    muns = ["Todos"] + sorted(df["nombre_municipio"].dropna().unique())
    sel_mun = st.sidebar.selectbox("Municipio", muns)

    # 2) Estrato (filtrado por municipio)
    df_mun = df if sel_mun == "Todos" else df[df["nombre_municipio"] == sel_mun]
    estratos = ["Todos"] + sorted(df_mun["estrato"].dropna().unique())
    sel_est = st.sidebar.selectbox("Estrato", estratos)

    # 3) Sexo víctima (filtrado por municipio + estrato)
    df_est = df_mun if sel_est == "Todos" else df_mun[df_mun["estrato"] == sel_est]
    sex_vict = ["Todos"] + sorted(df_est["sexo_victima"].dropna().unique())
    sel_sex_vict = st.sidebar.selectbox("Sexo de la víctima", sex_vict)

    # 4) Sexo agresor (filtrado por los tres anteriores)
    df_vict = (
        df_est
        if sel_sex_vict == "Todos"
        else df_est[df_est["sexo_victima"] == sel_sex_vict]
    )
    sex_agr = ["Todos"] + sorted(df_vict["sexo_agresor"].dropna().unique())
    sel_sex_agr = st.sidebar.selectbox("Sexo del agresor", sex_agr)

    # Horizonte de predicción (en meses, porque la serie es mensual)
    meses = st.sidebar.slider("Meses a pronosticar", 3, 60, 24, step=3)

    if st.sidebar.button("Generar Pronóstico"):
        try:
            # Entrena y obtiene la serie histórica (mensual)
            model, ts = train_model_for(
                sel_mun, sel_est, sel_sex_vict, sel_sex_agr
            )

            # Forecast
            future = model.make_future_dataframe(periods=meses, freq="MS")
            fcst = model.predict(future)

            # Separa histórico y pronóstico futuro
            last_hist = ts["ds"].max()
            fcst_future = fcst[fcst["ds"] > last_hist].copy()

            if fcst_future.empty:
                st.warning(
                    "El horizonte de pronóstico seleccionado no genera meses futuros."
                )
                return

            # No negativos
            fcst_future["yhat"] = fcst_future["yhat"].clip(lower=0)
            fcst_future["yhat_lower"] = fcst_future["yhat_lower"].clip(lower=0)
            fcst_future["yhat_upper"] = fcst_future["yhat_upper"].clip(lower=0)

            # === SUAVIZAR HISTÓRICO (PROMEDIO MÓVIL 3 MESES) ===
            ts_plot = ts.copy()
            ts_plot["y_smooth"] = (
                ts_plot["y"]
                .rolling(window=3, center=True, min_periods=1)
                .mean()
            )

            # Construcción de título dinámico (similar al legacy)
            filtros_detalle = []
            if sel_est != "Todos":
                filtros_detalle.append(f"Estrato {sel_est}")
            if sel_sex_vict != "Todos":
                filtros_detalle.append(f"Víctimas sexo {sel_sex_vict}")
            if sel_sex_agr != "Todos":
                filtros_detalle.append(f"Agresores sexo {sel_sex_agr}")

            primary = (
                "Casos de violencia"
                if not filtros_detalle
                else "Casos de violencia – " + ", ".join(filtros_detalle)
            )
            secondary = sel_mun if sel_mun != "Todos" else "Antioquia"
            chart_title = f"Predicción para {primary} en {secondary}"

            # ---------------------------
            #   GRÁFICO (ESTILO LEGACY)
            # ---------------------------
            fig = go.Figure()

            # Histórico suavizado (sin marcadores)
            fig.add_trace(
                go.Scatter(
                    x=ts_plot["ds"],
                    y=ts_plot["y_smooth"],
                    name="Histórico (prom. móvil 3 meses)",
                    mode="lines",
                )
            )

            # Pronóstico
            fig.add_trace(
                go.Scatter(
                    x=fcst_future["ds"],
                    y=fcst_future["yhat"],
                    name="Pronóstico",
                    mode="lines",
                )
            )

            # Intervalo de confianza
            fig.add_trace(
                go.Scatter(
                    x=pd.concat(
                        [fcst_future["ds"], fcst_future["ds"][::-1]]
                    ),
                    y=pd.concat(
                        [
                            fcst_future["yhat_upper"],
                            fcst_future["yhat_lower"][::-1],
                        ]
                    ),
                    fill="toself",
                    fillcolor="rgba(0,100,80,0.2)",
                    line=dict(color="rgba(0,0,0,0)"),
                    name="Intervalo de confianza",
                )
            )

            # Línea separadora entre histórico y futuro
            max_hist_y = float(ts_plot["y_smooth"].max()) * 1.1
            fig.add_shape(
                type="line",
                x0=last_hist,
                x1=last_hist,
                y0=0,
                y1=max_hist_y,
                line=dict(color="gray", dash="dash"),
            )

            fig.update_layout(
                title=chart_title,
                xaxis_title="Fecha",
                yaxis_title="Casos",
                legend_title="Series",
                autosize=True,
                margin=dict(l=40, r=40, t=80, b=40),
            )

            st.plotly_chart(fig, use_container_width=True)

        except ValueError as e:
            st.warning(str(e))



if __name__ == "__main__":
    main()
