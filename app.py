import pandas as pd
import streamlit as st
from prophet import Prophet
import plotly.graph_objects as go


@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    """Load data from an Excel file.

    Args:
        path (str): Path to the Excel file.

    Returns:
        pd.DataFrame: DataFrame containing the loaded data.
    """
    return pd.read_excel(path)


@st.cache_resource
def train_model_for(
    categoria: str,
    subregion: str,
    municipio: str, 
    subcategoria: str) -> tuple[Prophet, pd.DataFrame]:
    """
    Filter data for a specific municipality and subcategory, then train a Prophet model.

    Args:
        categoria (str): Category of the data.
        subregion (str): Subregion of the data.
        municipio (str): Municipality to filter the data.
        subcategoria (str): Subcategory to filter the data.

    Returns:
        tuple[Prophet, pd.DataFrame]: A tuple containing the trained Prophet model and the filtered DataFrame.
    """
    df = load_data("Consolidado.xlsx")
    
    df_filt = df.copy()
    if categoria != "Todos":
        df_filt = df_filt[df_filt["categoria"] == categoria]
    if subregion != "Todos":
        df_filt = df_filt[df_filt["subregion"] == subregion]
    if municipio != "Todos":
        df_filt = df_filt[df_filt["municipio"] == municipio]
    if subcategoria != "Todos":
        df_filt = df_filt[df_filt["subcategoria"] == subcategoria]
    
    ts = (
        df_filt
        .groupby("año", as_index=False)["casos"]
        .sum()
        .rename(columns={"casos": "y"})
    )
    
    ts["ds"] = pd.to_datetime(ts["año"].astype(str) + "-01-01")
    ts = ts[["ds", "y"]].sort_values("ds")
    
    model = Prophet(yearly_seasonality=True)
    model.fit(ts)
    
    return model


def main():
    st.title("Pronóstico de Casos de Violencia en el Departamento de Antioquia")
    st.sidebar.header("Filtros")

    df = load_data("Consolidado.xlsx")
    
    cats = ["Todos"] + sorted(df["categoria"].dropna().unique())
    sel_cat = st.sidebar.selectbox("Categoría", cats)

    df_cat = df if sel_cat == "Todos" else df[df["categoria"] == sel_cat]
    subregs = ["Todos"] + sorted(df_cat["subregion"].dropna().unique())
    sel_subreg = st.sidebar.selectbox("Subregión", subregs)

    df_subreg = df_cat if sel_subreg == "Todos" else df_cat[df_cat["subregion"] == sel_subreg]
    subcats = ["Todos"] + sorted(df_subreg["subcategoria"].dropna().unique())
    sel_subcat = st.sidebar.selectbox("Subcategoría", subcats)

    df_subcat = df_subreg if sel_subcat == "Todos" else df_subreg[df_subreg["subcategoria"] == sel_subcat]
    muns = ["Todos"] + sorted(df_subcat["municipio"].dropna().unique())
    sel_mun = st.sidebar.selectbox("Municipio", muns)

    years = st.sidebar.slider("Años a pronosticar", 1, 10, 5)

    if st.sidebar.button("Generar Pronóstico"):
        model, ts = train_model_for(sel_cat, sel_subreg, sel_subcat, sel_mun)
        future = model.make_future_dataframe(periods=years, freq="Y")
        fcst   = model.predict(future)

        last_hist  = ts["ds"].max()
        fcst_future = fcst[fcst["ds"] > last_hist]

        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=ts["ds"], y=ts["y"],
            name="Histórico", mode="lines+markers"
        ))
        
        fig.add_trace(go.Scatter(
            x=fcst_future["ds"], y=fcst_future["yhat"],
            name="Pronóstico", mode="lines"
        ))
        
        fig.add_trace(go.Scatter(
            x=pd.concat([fcst_future["ds"], fcst_future["ds"][::-1]]),
            y=pd.concat([fcst_future["yhat_upper"], fcst_future["yhat_lower"][::-1]]),
            fill="toself",
            fillcolor="rgba(0,100,80,0.2)",
            line=dict(color="rgba(0,0,0,0)"),
            name="Intervalo confianza"
        ))
        
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

        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()