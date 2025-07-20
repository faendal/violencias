import pandas as pd
import streamlit as st
from prophet import Prophet
import plotly.graph_objects as go

@st.cache_data
def load_data(path: str = "Consolidado.xlsx") -> pd.DataFrame:
    """Carga el Excel completo con todas las columnas necesarias."""
    return pd.read_excel(path)

@st.cache_resource
def train_model_for(
    categoria: str,
    subregion: str,
    subcategoria: str,
    municipio: str
) -> tuple[Prophet, pd.DataFrame]:
    """
    Filtra según los cuatro niveles (o 'Todos'), agrupa por año,
    entrena un Prophet y devuelve (modelo, ts) donde ts tiene columnas ['ds','y'].
    """
    df = load_data()
    df_f = df.copy()
    if categoria    != "Todos":
        df_f = df_f[df_f["categoria"]  == categoria]
    if subregion    != "Todos":
        df_f = df_f[df_f["subregion"]  == subregion]
    if subcategoria != "Todos":
        df_f = df_f[df_f["subcategoria"] == subcategoria]
    if municipio    != "Todos":
        df_f = df_f[df_f["municipio"]  == municipio]

    # Construimos la serie anual
    ts = (
        df_f
        .groupby("año", as_index=False)["casos"]
        .sum()
        .rename(columns={"casos": "y"})
    )
    ts["ds"] = pd.to_datetime(ts["año"].astype(str) + "-01-01")
    ts = ts[["ds", "y"]].sort_values("ds")

    model = Prophet(yearly_seasonality=True)
    model.fit(ts)
    return model, ts

def main():
    st.title("Forecast de Casos de Violencia")
    st.sidebar.header("Filtros")

    df = load_data()

    # 1) Categoría
    cats = ["Todos"] + sorted(df["categoria"].dropna().unique())
    sel_cat = st.sidebar.selectbox("Categoría", cats)

    # 2) Subregión (filtrada por categoría)
    df_cat = df if sel_cat == "Todos" else df[df["categoria"] == sel_cat]
    subregs = ["Todos"] + sorted(df_cat["subregion"].dropna().unique())
    sel_subreg = st.sidebar.selectbox("Subregión", subregs)

    # 3) Subcategoría (filtrada por categoría + subregión)
    df_subreg = (
        df_cat
        if sel_subreg == "Todos"
        else df_cat[df_cat["subregion"] == sel_subreg]
    )
    subcats = ["Todos"] + sorted(df_subreg["subcategoria"].dropna().unique())
    sel_subcat = st.sidebar.selectbox("Subcategoría", subcats)

    # 4) Municipio (filtrado por los tres anteriores)
    df_subcat = (
        df_subreg
        if sel_subcat == "Todos"
        else df_subreg[df_subreg["subcategoria"] == sel_subcat]
    )
    muns = ["Todos"] + sorted(df_subcat["municipio"].dropna().unique())
    sel_mun = st.sidebar.selectbox("Municipio", muns)

    # Horizonte de predicción
    years = st.sidebar.slider("Años a pronosticar", 1, 10, 5)

    if st.sidebar.button("Generar Pronóstico"):
        model, ts = train_model_for(sel_cat, sel_subreg, sel_subcat, sel_mun)

        future = model.make_future_dataframe(periods=years, freq="Y")
        fcst   = model.predict(future)

        # Separar histórico de forecast
        last_hist   = ts["ds"].max()
        fcst_future = fcst[fcst["ds"] > last_hist]

        # Construir gráfico
        fig = go.Figure()

        # a) Históricos
        fig.add_trace(go.Scatter(
            x=ts["ds"], y=ts["y"],
            name="Histórico", mode="lines+markers"
        ))

        # b) Pronóstico
        fig.add_trace(go.Scatter(
            x=fcst_future["ds"], y=fcst_future["yhat"],
            name="Pronóstico", mode="lines"
        ))

        # c) Intervalo de confianza
        fig.add_trace(go.Scatter(
            x=pd.concat([fcst_future["ds"], fcst_future["ds"][::-1]]),
            y=pd.concat([fcst_future["yhat_upper"], fcst_future["yhat_lower"][::-1]]),
            fill="toself",
            fillcolor="rgba(0,100,80,0.2)",
            line=dict(color="rgba(0,0,0,0)"),
            name="Intervalo confianza"
        ))

        # d) Línea separadora
        fig.add_shape(
            type="line",
            x0=last_hist, x1=last_hist,
            y0=0, y1=ts["y"].max() * 1.1,
            line=dict(color="gray", dash="dash")
        )

        fig.update_layout(
            title=f"{sel_cat} / {sel_subreg} / {sel_subcat} / {sel_mun}",
            xaxis_title="Fecha",
            yaxis_title="Casos",
            legend_title="Series",
            margin=dict(l=40, r=40, t=60, b=40)
        )

        # Ajuste automático al ancho del canvas
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()