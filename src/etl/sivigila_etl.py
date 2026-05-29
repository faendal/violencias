from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pandas as pd
import unicodedata

# Rutas base
DATA_RAW = Path("data/raw")
DATA_PROCESSED = Path("data/processed")

MAP_NATURALEZA = {
    "1": "Violencia Física",
    "2": "Violencia Psicológica",
    "3": "Negligencia y Abandono",
}

MAP_VIO_SEXUAL = {
    "5": "Acoso sexual",
    "6": "Acceso carnal",
    "7": "Explotación sexual",
    "10": "Trata de personas",
    "12": "Actos sexuales",
    "14": "Otras violencias sexuales",
    "15": "Mutilación Genital",
}

MONTH_NAME_ES: Dict[int, str] = {
    1: "ENERO",
    2: "FEBRERO",
    3: "MARZO",
    4: "ABRIL",
    5: "MAYO",
    6: "JUNIO",
    7: "JULIO",
    8: "AGOSTO",
    9: "SEPTIEMBRE",
    10: "OCTUBRE",
    11: "NOVIEMBRE",
    12: "DICIEMBRE",
}


def normalize_text(value: str) -> str:
    """Normaliza texto: quita tildes, deja mayúsculas y espacios simples."""
    if value is None:
        return ""
    text = str(value).strip().upper()
    text = "".join(
        c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c)
    )
    while "  " in text:
        text = text.replace("  ", " ")
    return text


def capitalize_text(value: str) -> str:
    """Convierte texto a Title Case."""
    if value is None:
        return ""
    txt = str(value).strip().lower()
    return txt.title()


def generate_sequential_ids(n: int, prefix: str, width: int = 6) -> List[str]:
    """Genera IDs tipo PREF_000001, PREF_000002, ..."""
    return [f"{prefix}{i:0{width}d}" for i in range(1, n + 1)]


def load_dane_municipios(path: Path) -> pd.DataFrame:
    """Carga el maestro DANE (municipios)."""
    if not path.exists():
        raise FileNotFoundError(f"No se encontró municipios_dane.xlsx en: {path}")

    df = pd.read_excel(
        path,
        dtype={
            "cod_dpto": str,
            "cod_mpio": str,
            "cod_mpio_dane": str,
        },
    )

    df["cod_dpto"] = df["cod_dpto"].astype(str).str.zfill(2)
    df["cod_mpio"] = df["cod_mpio"].astype(str).str.zfill(3)
    df["cod_mpio_dane"] = df["cod_dpto"] + df["cod_mpio"]

    if "nom_dpto_norm" not in df.columns:
        df["nom_dpto_norm"] = df["nom_dpto"].apply(normalize_text)
    if "nom_mpio_norm" not in df.columns:
        df["nom_mpio_norm"] = df["nom_mpio"].apply(normalize_text)

    df = df.drop_duplicates(subset=["cod_mpio_dane"])
    return df


def categorize_age(df: pd.DataFrame, col: str = "edad_") -> pd.DataFrame:
    """Clasifica la edad numérica en rangos predefinidos."""
    if col not in df.columns:
        return df

    edad_num = pd.to_numeric(df[col], errors="coerce")

    bins = [-1, 2, 7, 12, 18, 24, 34, 44, 54, 65, 200]
    labels = [
        "0-2",
        "2-7",
        "7-12",
        "12-18",
        "18-24",
        "25-34",
        "35-44",
        "45-54",
        "54-65",
        "65+",
    ]

    df["rango_edad"] = pd.cut(edad_num, bins=bins, labels=labels, right=True).astype(
        str
    )

    df["rango_edad"] = df["rango_edad"].replace("nan", "Sin Dato")
    return df


def load_sivigila(path: Path) -> pd.DataFrame:
    """Carga el CSV de SIVIGILA usando fec_not como fecha principal."""
    if not path.exists():
        raise FileNotFoundError(f"No se encontró sivigila_violencia.csv en: {path}")

    df = pd.read_csv(path, dtype=str, low_memory=False)

    df["fec_not"] = pd.to_datetime(df["fec_not"], errors="coerce", dayfirst=True)
    if "fec_hecho" in df.columns:
        df["fec_hecho"] = pd.to_datetime(
            df["fec_hecho"], errors="coerce", dayfirst=True
        )

    mask_rango = (df["fec_not"] >= "2013-01-01") & (df["fec_not"] <= "2022-12-09")
    df = df[mask_rango].copy()

    df = categorize_age(df)

    if "cod_dpto_o" in df.columns:
        df["cod_dpto_o"] = df["cod_dpto_o"].astype(str).str.zfill(2)
    if "cod_mun_o" in df.columns:
        df["cod_mun_o"] = df["cod_mun_o"].astype(str).str.zfill(3)

    return df


def filter_antioquia_and_map_municipios(
    df_siv: pd.DataFrame,
    df_dane: pd.DataFrame,
) -> pd.DataFrame:
    """Filtra casos ocurridos en Antioquia y mapea municipio DANE."""

    if "cod_dpto_o" not in df_siv.columns or "cod_mun_o" not in df_siv.columns:
        raise KeyError("Faltan columnas 'cod_dpto_o' o 'cod_mun_o' en SIVIGILA.")

    df_siv["cod_dpto_o"] = df_siv["cod_dpto_o"].astype(str).str.zfill(2)
    df_siv["cod_mun_o"] = df_siv["cod_mun_o"].astype(str).str.zfill(3)

    df_ant = df_siv[df_siv["cod_dpto_o"] == "05"].copy()
    print(f"Registros totales SIVIGILA: {len(df_siv):,}")
    print(f"Registros OCURRIDOS en Antioquia: {len(df_ant):,}")

    df_dane_ant = df_dane[df_dane["cod_dpto"] == "05"].copy()

    df_merged = df_ant.merge(
        df_dane_ant[["cod_dpto", "nom_dpto", "cod_mpio", "nom_mpio", "cod_mpio_dane"]],
        how="left",
        left_on=["cod_dpto_o", "cod_mun_o"],
        right_on=["cod_dpto", "cod_mpio"],
        suffixes=("", "_dane"),
    )

    df_merged["cod_mpio_resi"] = df_merged["cod_mpio_dane"]
    df_merged["nom_mpio_resi"] = df_merged["nom_mpio"]

    df_merged["nom_mpio_resi"] = df_merged["nom_mpio_resi"].fillna("SIN MAPEAR")

    return df_merged


def build_dim_tiempo(df: pd.DataFrame) -> pd.DataFrame:
    """Dimensión de tiempo basada en la fecha de notificación."""
    dim = df[["fec_not", "semana", "año", "periodo_epid", "mes_caso"]].copy()
    dim = dim.dropna(subset=["fec_not"])
    dim["fecha"] = pd.to_datetime(dim["fec_not"], errors="coerce")
    dim = dim.dropna(subset=["fecha"])
    dim = dim.sort_values("fecha").drop_duplicates(subset=["fecha"])

    dim["anio_hecho"] = dim["fecha"].dt.year
    dim["mes_num"] = dim["fecha"].dt.month
    dim["nombre_mes"] = dim["mes_num"].map(MONTH_NAME_ES)
    dim["trimestre"] = ((dim["mes_num"] - 1) // 3) + 1

    dim = dim.reset_index(drop=True)
    dim["id_tiempo"] = generate_sequential_ids(len(dim), prefix="TIME_")

    dim = dim[
        ["id_tiempo", "fecha", "anio_hecho", "mes_num", "nombre_mes", "trimestre"]
    ]
    return dim


def build_dim_municipio(df: pd.DataFrame) -> pd.DataFrame:
    """DimMunicipio (usando el nombre mapeado)."""
    dim = (
        df[["cod_mpio_resi", "nom_mpio_resi"]]
        .dropna(subset=["cod_mpio_resi"])
        .drop_duplicates()
        .copy()
    )
    dim["cod_mpio_dane"] = dim["cod_mpio_resi"].astype(str).str.zfill(5)
    dim["nombre_municipio"] = dim["nom_mpio_resi"].apply(capitalize_text)

    dim = dim.sort_values("cod_mpio_dane").reset_index(drop=True)
    dim["id_municipio"] = dim["cod_mpio_dane"].apply(lambda x: f"MUN_{x}")

    return dim[["id_municipio", "cod_mpio_dane", "nombre_municipio"]]


def build_dim_generic(
    df: pd.DataFrame, cols: List[str], id_col: str, prefix: str
) -> pd.DataFrame:
    """Crea una dimensión genérica."""
    existing = [c for c in cols if c in df.columns]
    dim = df[existing].drop_duplicates().reset_index(drop=True)
    dim[id_col] = generate_sequential_ids(len(dim), prefix=prefix)
    return dim[[id_col] + existing]


def build_dim_evento(df: pd.DataFrame) -> pd.DataFrame:
    """Construye DimEvento y TRADUCE los códigos a texto."""
    cols = [
        "naturaleza",
        "nat_viosex",
        "ambito_lug",
        "escenario",
        "actividad",
        "orient_sex",
        "ident_gene",
        "consum_spa",
        "mujer_cabf",
        "antec",
        "sust_vict",
        "tip_cas_",
        "fuente",
    ]

    existing = [c for c in cols if c in df.columns]
    dim = df[existing].copy()

    if "naturaleza" in dim.columns:
        dim["naturaleza"] = (
            dim["naturaleza"].astype(str).str.replace(r"\.0$", "", regex=True)
        )
        dim["naturaleza"] = (
            dim["naturaleza"].map(MAP_NATURALEZA).fillna("Otro/Desconocido")
        )

    if "nat_viosex" in dim.columns:
        dim["nat_viosex"] = (
            dim["nat_viosex"].astype(str).str.replace(r"\.0$", "", regex=True)
        )
        dim["nat_viosex"] = dim["nat_viosex"].map(MAP_VIO_SEXUAL).fillna("No Aplica")

    dim = dim.drop_duplicates().reset_index(drop=True)
    dim["id_evento"] = generate_sequential_ids(len(dim), prefix="EVN_")

    return dim[["id_evento"] + existing]


def build_dim_victima(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "sexo_",
        "grupo_edad_quinque",
        "estrato_",
        "ocupacion_",
        "per_etn_",
        "tip_ss_",
        "rango_edad",
    ]
    return build_dim_generic(df, cols, "id_victima", "VIC_")


def build_dim_agresor(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["sexo_agre", "edad_agre", "conv_agre", "r_fam_vic", "armas"]
    return build_dim_generic(df, cols, "id_agresor", "AGR_")


def build_dim_acciones(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["ac_mental", "remit_prot", "inf_aut"]
    return build_dim_generic(df, cols, "id_accion", "ACC_")


def build_fact_hechos(
    df_ant: pd.DataFrame,
    dim_tiempo: pd.DataFrame,
    dim_mpio: pd.DataFrame,
    dim_victima: pd.DataFrame,
    dim_evento: pd.DataFrame,
    dim_agresor: pd.DataFrame,
) -> pd.DataFrame:
    """Construye la tabla de hechos."""
    df_fact = df_ant.copy()

    dt = dim_tiempo[["id_tiempo", "fecha"]].copy()
    df_fact = df_fact.merge(dt, how="left", left_on="fec_not", right_on="fecha").drop(
        columns=["fecha"]
    )

    df_fact["cod_mpio_resi"] = df_fact["cod_mpio_resi"].astype(str).str.zfill(5)
    dm = dim_mpio[["id_municipio", "cod_mpio_dane"]].copy()
    df_fact = df_fact.merge(
        dm, how="left", left_on="cod_mpio_resi", right_on="cod_mpio_dane"
    )

    if "naturaleza" in df_fact.columns:
        df_fact["naturaleza"] = (
            df_fact["naturaleza"].astype(str).str.replace(r"\.0$", "", regex=True)
        )
        df_fact["naturaleza"] = (
            df_fact["naturaleza"].map(MAP_NATURALEZA).fillna("Otro/Desconocido")
        )

    if "nat_viosex" in df_fact.columns:
        df_fact["nat_viosex"] = (
            df_fact["nat_viosex"].astype(str).str.replace(r"\.0$", "", regex=True)
        )
        df_fact["nat_viosex"] = (
            df_fact["nat_viosex"].map(MAP_VIO_SEXUAL).fillna("No Aplica")
        )

    join_cols = [c for c in dim_evento.columns if c != "id_evento"]
    df_fact = df_fact.merge(dim_evento, how="left", on=join_cols)

    v_cols = [
        c for c in dim_victima.columns if c != "id_victima" and c in df_fact.columns
    ]
    df_fact = df_fact.merge(dim_victima, how="left", on=v_cols)

    g_cols = [
        c for c in dim_agresor.columns if c != "id_agresor" and c in df_fact.columns
    ]
    df_fact = df_fact.merge(dim_agresor, how="left", on=g_cols)

    df_fact["casos"] = 1
    df_fact["id_hecho"] = generate_sequential_ids(len(df_fact), prefix="HEC_")

    fact_cols = [
        "id_hecho",
        "id_tiempo",
        "id_municipio",
        "id_victima",
        "id_evento",
        "id_agresor",
        "id_accion",
        "casos",
        "fec_not",
    ]
    final_cols = [c for c in fact_cols if c in df_fact.columns]

    return df_fact[final_cols]


def build_serie_mensual_consolidado(
    fact: pd.DataFrame,
    dim_tiempo: pd.DataFrame,
    dim_mpio: pd.DataFrame,
    dim_victima: pd.DataFrame,
    dim_evento: pd.DataFrame,
    dim_agresor: pd.DataFrame,
) -> pd.DataFrame:
    """Construye el consolidado mensual con textos descriptivos."""

    ft = (
        fact.merge(
            dim_tiempo[["id_tiempo", "fecha", "anio_hecho", "mes_num"]],
            on="id_tiempo",
            how="left",
        )
        .merge(
            dim_mpio[["id_municipio", "nombre_municipio"]],
            on="id_municipio",
            how="left",
        )
        .merge(
            dim_victima[["id_victima", "sexo_", "estrato_", "rango_edad"]],
            on="id_victima",
            how="left",
        )
        .merge(dim_agresor[["id_agresor", "sexo_agre"]], on="id_agresor", how="left")
        .merge(
            dim_evento[["id_evento", "naturaleza", "nat_viosex"]],
            on="id_evento",
            how="left",
        )
    )

    ft = ft.dropna(subset=["fecha"])
    ft["anio_hecho"] = ft["anio_hecho"].astype(int)
    ft["mes_num"] = ft["mes_num"].astype(int)

    # Renombres
    ft["sexo_victima"] = ft["sexo_"]
    ft["sexo_agresor"] = ft["sexo_agre"]
    ft["estrato"] = ft["estrato_"].fillna("Sin Dato")
    ft["rango_edad"] = ft["rango_edad"].fillna("Sin Dato")

    ft["naturaleza"] = ft["naturaleza"].fillna("Otro/Desconocido")
    ft["nat_viosex"] = ft["nat_viosex"].fillna("No Aplica")

    group_cols = [
        "fecha",
        "anio_hecho",
        "mes_num",
        "nombre_municipio",
        "estrato",
        "sexo_victima",
        "sexo_agresor",
        "naturaleza",
        "nat_viosex",
        "rango_edad",
    ]

    serie = ft.groupby(group_cols, as_index=False)["casos"].sum()
    serie["anio_mes"] = (
        serie["anio_hecho"].astype(str)
        + "-"
        + serie["mes_num"].astype(str).str.zfill(2)
    )

    return serie.sort_values(["nombre_municipio", "fecha"])


def main() -> None:
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    dane_path = DATA_RAW / "municipios_dane.xlsx"
    siv_path = DATA_RAW / "sivigila_violencia.csv"

    df_dane = load_dane_municipios(dane_path)
    df_siv = load_sivigila(siv_path)

    df_ant = filter_antioquia_and_map_municipios(df_siv, df_dane)

    dim_tiempo = build_dim_tiempo(df_ant)
    dim_mpio = build_dim_municipio(df_ant)
    dim_victima = build_dim_victima(df_ant)
    dim_agresor = build_dim_agresor(df_ant)
    dim_acciones = build_dim_acciones(df_ant)
    dim_evento = build_dim_evento(df_ant)

    fact_hechos = build_fact_hechos(
        df_ant, dim_tiempo, dim_mpio, dim_victima, dim_evento, dim_agresor
    )

    serie_mensual = build_serie_mensual_consolidado(
        fact_hechos, dim_tiempo, dim_mpio, dim_victima, dim_evento, dim_agresor
    )

    excel_path = DATA_PROCESSED / "SIVIGILA_Violencias_Modelo.xlsx"
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        dim_tiempo.to_excel(writer, sheet_name="DimTiempo", index=False)
        dim_mpio.to_excel(writer, sheet_name="DimMunicipio", index=False)
        dim_victima.to_excel(writer, sheet_name="DimVictima", index=False)
        dim_evento.to_excel(writer, sheet_name="DimEvento", index=False)
        fact_hechos.to_excel(writer, sheet_name="FactHechos", index=False)
        serie_mensual.to_excel(writer, sheet_name="SerieMensual", index=False)

    serie_mensual.to_csv(DATA_PROCESSED / "SerieMensual.csv", index=False)

    print(f"Proceso completado. Modelo en {excel_path}")
    print("Serie mensual en data/processed/SerieMensual.csv")


if __name__ == "__main__":
    main()
