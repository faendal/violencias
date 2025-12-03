from pathlib import Path
from typing import Tuple

import pandas as pd
import streamlit as st
from prophet import Prophet
import plotly.graph_objects as go

# Ruta al consolidado mensual
DATA_PATH = Path("data/processed/SerieMensual.csv")


@st.cache_data
def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Carga la serie mensual consolidada."""
    df = pd.read_csv(path, parse_dates=["fecha"])

    # Tipos básicos
    df["anio_hecho"] = df["anio_hecho"].astype(int)
    df["mes_num"] = df["mes_num"].astype(int)

    # Columnas de texto
    text_cols = [
        "nombre_municipio",
        "estrato",
        "sexo_victima",
        "sexo_agresor",
        "naturaleza",
        "nat_viosex",
    ]
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].astype(str)
        else:
            df[col] = "Sin Dato"

    return df


@st.cache_resource
def train_model_for(
    municipio: str,
    estrato: str,
    sexo_victima: str,
    sexo_agresor: str,
    naturaleza: str,
    nat_viosex: str,
) -> Tuple[Prophet, pd.DataFrame]:

    df = load_data()
    df_f = df.copy()

    # Filtros
    if municipio != "Todos":
        df_f = df_f[df_f["nombre_municipio"] == municipio]
    if estrato != "Todos":
        df_f = df_f[df_f["estrato"] == estrato]
    if sexo_victima != "Todos":
        df_f = df_f[df_f["sexo_victima"] == sexo_victima]
    if sexo_agresor != "Todos":
        df_f = df_f[df_f["sexo_agresor"] == sexo_agresor]
    if naturaleza != "Todos":
        df_f = df_f[df_f["naturaleza"] == naturaleza]
    if nat_viosex != "Todos":
        df_f = df_f[df_f["nat_viosex"] == nat_viosex]

    if df_f.empty:
        raise ValueError("No hay datos para la combinación de filtros seleccionada.")

    # Agrupar por mes
    ts = (
        df_f.groupby("fecha", as_index=False)["casos"]
        .sum()
        .rename(columns={"fecha": "ds", "casos": "y"})
        .sort_values("ds")
    )

    if ts["y"].sum() == 0:
        raise ValueError("La serie resultante tiene 0 casos.")

    model = Prophet(
        yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False
    )
    model.fit(ts)

    return model, ts


def main() -> None:
    st.set_page_config(layout="wide", page_title="Pronóstico de Casos de Violencia")
    st.title("Pronóstico De Casos De Violencia En El Departamento De Antioquia")

    df = load_data()

    st.sidebar.header("Filtros")

    # Selectores (con orden alfabético para facilitar búsqueda)
    # 1. Municipio
    muns = ["Todos"] + sorted(df["nombre_municipio"].unique())
    sel_mun = st.sidebar.selectbox("Municipio (Ocurrencia)", muns)

    # Filtro cascada
    df_f = df if sel_mun == "Todos" else df[df["nombre_municipio"] == sel_mun]

    # 2. Naturaleza (Texto descriptivo)
    nats = ["Todos"] + sorted(df_f["naturaleza"].unique())
    sel_nat = st.sidebar.selectbox("Modalidad", nats)
    if sel_nat != "Todos":
        df_f = df_f[df_f["naturaleza"] == sel_nat]

    # 3. Violencia Sexual (Texto descriptivo)
    vios = ["Todos"] + sorted(df_f["nat_viosex"].unique())
    sel_vio = st.sidebar.selectbox("Tipo Violencia Sexual", vios)
    if sel_vio != "Todos":
        df_f = df_f[df_f["nat_viosex"] == sel_vio]

    # Otros filtros
    estratos = ["Todos"] + sorted(df_f["estrato"].unique())
    sel_est = st.sidebar.selectbox("Estrato", estratos)
    if sel_est != "Todos":
        df_f = df_f[df_f["estrato"] == sel_est]

    sex_vict = ["Todos"] + sorted(df_f["sexo_victima"].unique())
    sel_sex_vict = st.sidebar.selectbox("Sexo Víctima", sex_vict)
    if sel_sex_vict != "Todos":
        df_f = df_f[df_f["sexo_victima"] == sel_sex_vict]

    sex_agr = ["Todos"] + sorted(df_f["sexo_agresor"].unique())
    sel_sex_agr = st.sidebar.selectbox("Sexo Agresor", sex_agr)

    meses = st.sidebar.slider("Meses a pronosticar", 3, 60, 24, step=3)

    if st.sidebar.button("Generar Pronóstico"):
        try:
            model, ts = train_model_for(
                sel_mun, sel_est, sel_sex_vict, sel_sex_agr, sel_nat, sel_vio
            )

            future = model.make_future_dataframe(periods=meses, freq="MS")
            fcst = model.predict(future)

            last_hist = ts["ds"].max()
            fcst_future = fcst[fcst["ds"] > last_hist].copy()

            # Clip negativos
            cols_clip = ["yhat", "yhat_lower", "yhat_upper"]
            fcst_future[cols_clip] = fcst_future[cols_clip].clip(lower=0)

            # Histórico suavizado
            ts["y_smooth"] = ts["y"].rolling(3, center=True, min_periods=1).mean()

            # Titulo dinámico
            parts = []
            if sel_nat != "Todos":
                parts.append(sel_nat)
            if sel_vio != "Todos":
                parts.append(sel_vio)
            subtitle = ", ".join(parts) if parts else "General"

            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=ts["ds"],
                    y=ts["y_smooth"],
                    name="Histórico (suavizado)",
                    mode="lines",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=fcst_future["ds"],
                    y=fcst_future["yhat"],
                    name="Pronóstico",
                    mode="lines",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=pd.concat([fcst_future["ds"], fcst_future["ds"][::-1]]),
                    y=pd.concat(
                        [fcst_future["yhat_upper"], fcst_future["yhat_lower"][::-1]]
                    ),
                    fill="toself",
                    fillcolor="rgba(0,100,80,0.2)",
                    line=dict(color="rgba(0,0,0,0)"),
                    name="Confianza",
                )
            )

            fig.add_shape(
                type="line",
                x0=last_hist,
                x1=last_hist,
                y0=0,
                y1=ts["y_smooth"].max() * 1.1,
                line=dict(dash="dash", color="gray"),
            )

            fig.update_layout(
                title=f"Pronóstico: {subtitle} en {sel_mun}",
                xaxis_title="Fecha",
                yaxis_title="Casos",
            )
            st.plotly_chart(fig, use_container_width=True)

        except ValueError as e:
            st.error(f"Error: {e}")


if __name__ == "__main__":
    main()
