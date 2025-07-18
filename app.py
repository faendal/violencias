import pandas as pd
import streamlit as st
from prophet import Prophet
import plotly.graph_objects as go


@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    return pd.read_excel(path)


@st.cache_resource
def train_model_for(municipio: str, subcategoria: str) -> Prophet:
    df = load_data("Consolidado.xlsx")
    df_filt = df[(df["municipio"] == municipio) & (df["subcategoria"] == subcategoria)]
    ts = (
        df_filt.groupby("año", as_index=False)["casos"]
        .sum()
        .rename(columns={"casos": "y"})
    )
    ts["ds"] = pd.to_datetime(ts["año"].astype(str) + "-01-01")
    ts = ts[["ds", "y"]].sort_values("ds")
    model = Prophet(yearly_seasonality=True)
    model.fit(ts)
    return model


def main():
    st.title("Forecast de Casos de Violencia por Municipio y Subcategoría")
    st.sidebar.header("Filtros")

    df = load_data("Consolidado.xlsx")
    municipios = sorted(df["municipio"].unique())
    sel_mun = st.sidebar.selectbox("Municipio", municipios)

    subs = sorted(df[df["municipio"] == sel_mun]["subcategoria"].unique())
    sel_sub = st.sidebar.selectbox("Subcategoría", subs)

    years = st.sidebar.slider("Años a pronosticar", 1, 10, 5)

    if st.sidebar.button("Generar Pronóstico"):
        # Serie histórica
        df_filt = df[(df["municipio"] == sel_mun) & (df["subcategoria"] == sel_sub)]
        ts = (
            df_filt.groupby("año", as_index=False)["casos"]
            .sum()
            .rename(columns={"casos": "y"})
        )
        ts["ds"] = pd.to_datetime(ts["año"].astype(str) + "-01-01")
        ts = ts[["ds", "y"]].sort_values("ds")

        # Modelo y forecast
        model = train_model_for(sel_mun, sel_sub)
        future = model.make_future_dataframe(periods=years, freq="Y")
        fcst = model.predict(future)

        last_hist = ts["ds"].max()
        fcst_future = fcst[fcst["ds"] > last_hist]

        # Gráfico
        fig = go.Figure()
        # Histórico
        fig.add_trace(
            go.Scatter(x=ts["ds"], y=ts["y"], name="Histórico", mode="lines+markers")
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
        # Intervalo confianza
        fig.add_trace(
            go.Scatter(
                x=pd.concat([fcst_future["ds"], fcst_future["ds"][::-1]]),
                y=pd.concat(
                    [fcst_future["yhat_upper"], fcst_future["yhat_lower"][::-1]]
                ),
                fill="toself",
                fillcolor="rgba(0,100,80,0.2)",
                line=dict(color="rgba(0,0,0,0)"),
                showlegend=True,
                name="Intervalo confianza",
            )
        )
        # Línea separadora
        fig.add_shape(
            type="line",
            x0=last_hist,
            x1=last_hist,
            y0=0,
            y1=ts["y"].max() * 1.1,
            line=dict(color="gray", dash="dash"),
        )
        fig.update_layout(
            title=f"{sel_mun} − {sel_sub}",
            xaxis_title="Fecha",
            yaxis_title="Casos",
            legend_title="Series",
        )

        st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
