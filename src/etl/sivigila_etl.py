from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import unicodedata

# Rutas base
DATA_RAW = Path("data/raw")
DATA_PROCESSED = Path("data/processed")


# ----------------------------------------------------------------------
# Helpers generales
# ----------------------------------------------------------------------
def normalize_text(value: str) -> str:
    """Normaliza texto: quita tildes, deja mayúsculas y espacios simples."""
    if value is None:
        return ""
    text = str(value).strip().upper()
    text = "".join(
        c
        for c in unicodedata.normalize("NFKD", text)
        if not unicodedata.combining(c)
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


# ----------------------------------------------------------------------
# Carga de datos base
# ----------------------------------------------------------------------
def load_dane_municipios(path: Path) -> pd.DataFrame:
    """Carga el maestro DANE (municipios) generado desde el PDF."""
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


def load_sivigila(path: Path) -> pd.DataFrame:
    """Carga el CSV de SIVIGILA usando fec_not como fecha principal.

    - Lee el CSV.
    - Parsea fec_not (y fec_hecho por si la quieres luego como atributo).
    - Filtra el rango de fechas válido: 2013-01-01 a 2022-12-09.
    - Asegura códigos de dpto/mun residencia.
    """
    if not path.exists():
        raise FileNotFoundError(f"No se encontró sivigila_violencia.csv en: {path}")

    df = pd.read_csv(path, dtype=str, low_memory=False)

    # Fechas (usamos fec_not como referencia temporal principal)
    df["fec_not"] = pd.to_datetime(df["fec_not"], errors="coerce")
    if "fec_hecho" in df.columns:
        df["fec_hecho"] = pd.to_datetime(df["fec_hecho"], errors="coerce")

    # Filtrar rango de interés según fec_not
    mask_rango = (df["fec_not"] >= "2013-01-01") & (df["fec_not"] <= "2022-12-09")
    df = df[mask_rango].copy()

    # Códigos Divipola de residencia
    if "cod_dpto_r" in df.columns:
        df["cod_dpto_r"] = df["cod_dpto_r"].astype(str).str.zfill(2)
    if "cod_mun_r" in df.columns:
        df["cod_mun_r"] = df["cod_mun_r"].astype(str).str.zfill(3)

    # Año epidemiológico como numérico
    if "año" in df.columns:
        df["año"] = pd.to_numeric(df["año"], errors="coerce")

        # (Opcional pero recomendado) coherencia año vs fec_not
        year_from_date = df["fec_not"].dt.year
        mismatch = (df["año"] != year_from_date).sum()
        if mismatch > 0:
            print(f"Advertencia: {mismatch} registros con año != año(fec_not)")

    return df



# ----------------------------------------------------------------------
# Filtro a Antioquia y mapeo municipal
# ----------------------------------------------------------------------
def filter_antioquia_and_map_municipios(
    df_siv: pd.DataFrame,
    df_dane: pd.DataFrame,
) -> pd.DataFrame:
    """Filtra casos a Antioquia (cod_dpto_r == '05') y mapea municipio DANE."""

    if "cod_dpto_r" not in df_siv.columns or "cod_mun_r" not in df_siv.columns:
        raise KeyError("Faltan columnas 'cod_dpto_r' o 'cod_mun_r' en SIVIGILA.")

    # Filtro a Antioquia
    df_siv["cod_dpto_r"] = df_siv["cod_dpto_r"].astype(str).str.zfill(2)
    df_siv["cod_mun_r"] = df_siv["cod_mun_r"].astype(str).str.zfill(3)

    df_ant = df_siv[df_siv["cod_dpto_r"] == "05"].copy()
    print(f"Registros totales SIVIGILA: {len(df_siv):,}")
    print(f"Registros en Antioquia (cod_dpto_r == '05'): {len(df_ant):,}")

    # Maestro DANE solo Antioquia
    df_dane_ant = df_dane[df_dane["cod_dpto"] == "05"].copy()

    # Merge por códigos Divipola de residencia
    df_merged = df_ant.merge(
        df_dane_ant[
            ["cod_dpto", "nom_dpto", "cod_mpio", "nom_mpio", "cod_mpio_dane"]
        ],
        how="left",
        left_on=["cod_dpto_r", "cod_mun_r"],
        right_on=["cod_dpto", "cod_mpio"],
        suffixes=("", "_dane"),
    )

    # Campos geográficos de residencia oficiales
    df_merged["cod_dpto_resi"] = df_merged["cod_dpto_r"]
    df_merged["nom_dpto_resi"] = df_merged["nom_dpto"].fillna("ANTIOQUIA")
    df_merged["cod_mpio_resi"] = df_merged["cod_mpio_dane"]
    df_merged["nom_mpio_resi"] = df_merged["nom_mpio"]

    # Conteo de municipios sin mapear
    unmatched = df_merged["cod_mpio_resi"].isna().sum()
    print(f"Registros de Antioquia SIN municipio DANE mapeado: {unmatched:,}")

    return df_merged


# ----------------------------------------------------------------------
# Construcción de dimensiones
# ----------------------------------------------------------------------
def build_dim_tiempo(df: pd.DataFrame) -> pd.DataFrame:
    """Dimensión de tiempo basada en la fecha de notificación (fec_not)."""
    if "fec_not" not in df.columns:
        raise KeyError("No se encontró la columna 'fec_not' en SIVIGILA.")

    dim = df[["fec_not", "semana", "año", "periodo_epid", "mes_caso"]].copy()
    dim = dim.dropna(subset=["fec_not"])

    dim["fecha"] = pd.to_datetime(dim["fec_not"], errors="coerce")
    dim = dim.dropna(subset=["fecha"])

    dim = dim.sort_values("fecha").drop_duplicates(subset=["fecha"])

    dim["anio_hecho"] = dim["fecha"].dt.year
    dim["mes_num"] = dim["fecha"].dt.month
    dim["nombre_mes"] = dim["mes_num"].map(MONTH_NAME_ES)
    dim["trimestre"] = ((dim["mes_num"] - 1) // 3) + 1

    dim = dim.rename(columns={"año": "anio_epi", "semana": "semana_epi"})

    dim = dim.reset_index(drop=True)
    dim["id_tiempo"] = generate_sequential_ids(len(dim), prefix="TIME_")

    dim = dim[
        [
            "id_tiempo",
            "fecha",       
            "anio_hecho",   
            "mes_num",
            "nombre_mes",
            "trimestre",
            "semana_epi",
            "anio_epi",
            "periodo_epid",
            "mes_caso",
        ]
    ]

    return dim



def build_dim_departamento(df: pd.DataFrame) -> pd.DataFrame:
    """DimDepartamento (en este caso solo Antioquia)."""
    dim = (
        df[["cod_dpto_resi", "nom_dpto_resi"]]
        .dropna(subset=["cod_dpto_resi"])
        .drop_duplicates()
        .copy()
    )

    dim["cod_dpto_dane"] = dim["cod_dpto_resi"].astype(str).str.zfill(2)
    dim["nombre_departamento"] = dim["nom_dpto_resi"].apply(capitalize_text)

    dim = dim.drop(columns=["cod_dpto_resi", "nom_dpto_resi"])
    dim = dim.sort_values("cod_dpto_dane").reset_index(drop=True)

    dim["id_departamento"] = dim["cod_dpto_dane"].apply(lambda x: f"DEP_{x}")

    dim = dim[["id_departamento", "cod_dpto_dane", "nombre_departamento"]]
    return dim


def build_dim_municipio(df: pd.DataFrame) -> pd.DataFrame:
    """DimMunicipio (solo para Antioquia, usando cod_mpio_resi ya mapeado)."""
    dim = (
        df[["cod_mpio_resi", "nom_mpio_resi", "cod_dpto_resi"]]
        .dropna(subset=["cod_mpio_resi"])
        .drop_duplicates()
        .copy()
    )

    dim["cod_mpio_dane"] = dim["cod_mpio_resi"].astype(str).str.zfill(5)
    dim["cod_dpto_dane"] = dim["cod_dpto_resi"].astype(str).str.zfill(2)
    dim["nombre_municipio"] = dim["nom_mpio_resi"].apply(capitalize_text)

    dim["id_municipio"] = dim["cod_mpio_dane"].apply(lambda x: f"MUN_{x}")

    dim = dim[
        [
            "id_municipio",
            "cod_mpio_dane",
            "nombre_municipio",
            "cod_dpto_dane",
        ]
    ].sort_values("cod_mpio_dane")

    return dim


def build_dim_comuna_barrio(df: pd.DataFrame) -> pd.DataFrame:
    """Dimensión de comuna/barrio anclada al municipio de residencia."""
    cols = [
        "cod_mpio_resi",
        "nom_mpio_resi",
        "codigo_comuna",
        "nombre_comuna",
        "codigo_barrio",
        "nombre_barrio",
    ]
    existing = [c for c in cols if c in df.columns]

    if not existing:
        return pd.DataFrame()

    dim = df[existing].copy()

    dim = dim[
        dim.get("codigo_comuna").notna()
        | dim.get("codigo_barrio").notna()
    ]

    dim["cod_mpio_dane"] = dim["cod_mpio_resi"].astype(str).str.zfill(5)
    dim["nombre_municipio"] = dim["nom_mpio_resi"].apply(capitalize_text)

    if "nombre_comuna" in dim.columns:
        dim["nombre_comuna"] = dim["nombre_comuna"].apply(capitalize_text)
    if "nombre_barrio" in dim.columns:
        dim["nombre_barrio"] = dim["nombre_barrio"].apply(capitalize_text)

    dim = dim.drop_duplicates().reset_index(drop=True)
    dim["id_comuna_barrio"] = generate_sequential_ids(len(dim), prefix="CBA_")

    ordered_cols = [
        "id_comuna_barrio",
        "cod_mpio_dane",
        "nombre_municipio",
    ]
    for c in ["codigo_comuna", "nombre_comuna", "codigo_barrio", "nombre_barrio"]:
        if c in dim.columns:
            ordered_cols.append(c)

    dim = dim[ordered_cols]
    return dim


def build_dim_generic(
    df: pd.DataFrame,
    cols: List[str],
    id_col: str,
    prefix: str,
) -> pd.DataFrame:
    """Crea una dimensión genérica por combinación de columnas."""
    existing = [c for c in cols if c in df.columns]
    dim = df[existing].drop_duplicates().reset_index(drop=True)
    dim[id_col] = generate_sequential_ids(len(dim), prefix=prefix)
    dim = dim[[id_col] + existing]
    return dim


def build_dim_victima(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "sexo_",
        "grupo_edad_quinque",
        "grupo_edad_ciclo",
        "per_etn_",
        "tip_ss_",
        "ocupacion_",
        "estrato_",
        "gp_discapa",
        "gp_desplaz",
        "gp_migrant",
        "gp_carcela",
        "gp_gestan",
        "gp_indigen",
        "gp_pobicbf",
        "gp_mad_com",
        "gp_desmovi",
        "gp_psiquia",
        "gp_vic_vio",
        "gp_otros",
    ]
    return build_dim_generic(df, cols, id_col="id_victima", prefix="VIC_")


def build_dim_evento(df: pd.DataFrame) -> pd.DataFrame:
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
    return build_dim_generic(df, cols, id_col="id_evento", prefix="EVN_")


def build_dim_agresor(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "sexo_agre",
        "edad_agre",
        "r_fam_vic",
        "conv_agre",
        "r_nofiliar",
        "zona_conf",
        "armas",
    ]
    return build_dim_generic(df, cols, id_col="id_agresor", prefix="AGR_")


def build_dim_acciones(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "sp_its",
        "prof_hep_b",
        "prof_otras",
        "ac_anticon",
        "ac_ive",
        "ac_mental",
        "remit_prot",
        "inf_aut",
        "evi_mlegal",
    ]
    return build_dim_generic(df, cols, id_col="id_accion", prefix="ACC_")


# ----------------------------------------------------------------------
# Tabla de hechos y serie mensual
# ----------------------------------------------------------------------
VICT_COLS = [
    "sexo_",
    "grupo_edad_quinque",
    "grupo_edad_ciclo",
    "per_etn_",
    "tip_ss_",
    "ocupacion_",
    "estrato_",
    "gp_discapa",
    "gp_desplaz",
    "gp_migrant",
    "gp_carcela",
    "gp_gestan",
    "gp_indigen",
    "gp_pobicbf",
    "gp_mad_com",
    "gp_desmovi",
    "gp_psiquia",
    "gp_vic_vio",
    "gp_otros",
]

EVENTO_COLS = [
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

AGRESOR_COLS = [
    "sexo_agre",
    "edad_agre",
    "r_fam_vic",
    "conv_agre",
    "r_nofiliar",
    "zona_conf",
    "armas",
]

ACCION_COLS = [
    "sp_its",
    "prof_hep_b",
    "prof_otras",
    "ac_anticon",
    "ac_ive",
    "ac_mental",
    "remit_prot",
    "inf_aut",
    "evi_mlegal",
]


def build_fact_hechos(
    df_ant: pd.DataFrame,
    dim_tiempo: pd.DataFrame,
    dim_mpio: pd.DataFrame,
    dim_cb: pd.DataFrame,
    dim_victima: pd.DataFrame,
    dim_evento: pd.DataFrame,
    dim_agresor: pd.DataFrame,
    dim_acciones: pd.DataFrame,
) -> pd.DataFrame:
    """Construye la tabla de hechos (una fila por caso notificado)."""
    df_fact = df_ant.copy()

    # FK tiempo: usamos fec_not
    dt = dim_tiempo[["id_tiempo", "fecha"]].copy()
    df_fact = df_fact.merge(
        dt,
        how="left",
        left_on="fec_not",
        right_on="fecha",
    ).drop(columns=["fecha"])

    # FK municipio
    df_fact["cod_mpio_resi"] = df_fact["cod_mpio_resi"].astype(str).str.zfill(5)
    dim_mpio2 = dim_mpio[["id_municipio", "cod_mpio_dane"]].copy()
    df_fact = df_fact.merge(
        dim_mpio2,
        how="left",
        left_on="cod_mpio_resi",
        right_on="cod_mpio_dane",
    )

    # FK comuna/barrio
    if dim_cb is not None and not dim_cb.empty:
        join_cb_cols = [
            c for c in ["codigo_comuna", "codigo_barrio"] if c in df_fact.columns
        ]
        if join_cb_cols:
            dim_cb2 = dim_cb[
                ["id_comuna_barrio", "cod_mpio_dane"] + join_cb_cols
            ].copy()
            df_fact = df_fact.merge(
                dim_cb2,
                how="left",
                left_on=["cod_mpio_resi"] + join_cb_cols,
                right_on=["cod_mpio_dane"] + join_cb_cols,
            )

    # FK víctima
    v_cols = [c for c in VICT_COLS if c in df_fact.columns]
    dim_v2 = dim_victima[["id_victima"] + v_cols].copy()
    df_fact = df_fact.merge(dim_v2, how="left", on=v_cols)

    # FK evento
    e_cols = [c for c in EVENTO_COLS if c in df_fact.columns]
    dim_e2 = dim_evento[["id_evento"] + e_cols].copy()
    df_fact = df_fact.merge(dim_e2, how="left", on=e_cols)

    # FK agresor
    g_cols = [c for c in AGRESOR_COLS if c in df_fact.columns]
    dim_g2 = dim_agresor[["id_agresor"] + g_cols].copy()
    df_fact = df_fact.merge(dim_g2, how="left", on=g_cols)

    # FK acciones
    a_cols = [c for c in ACCION_COLS if c in df_fact.columns]
    dim_a2 = dim_acciones[["id_accion"] + a_cols].copy()
    df_fact = df_fact.merge(dim_a2, how="left", on=a_cols)

    # ID de hecho y medida
    df_fact = df_fact.reset_index(drop=True)
    df_fact["id_hecho"] = generate_sequential_ids(len(df_fact), prefix="HEC_")
    df_fact["casos"] = 1

    fact_cols = [
        "id_hecho",
        "id_tiempo",
        "id_municipio",
        "id_comuna_barrio",
        "id_victima",
        "id_evento",
        "id_agresor",
        "id_accion",
        "casos",
        "fec_not",
        "fec_hecho"
        "año",
        "semana",
        "periodo_epid",
        "mes_caso",
        "cod_dpto_resi",
        "cod_mpio_resi",
    ]
    fact_cols = [c for c in fact_cols if c in df_fact.columns]

    fact = df_fact[fact_cols].copy()
    return fact


def build_serie_mensual_consolidado(
    fact: pd.DataFrame,
    dim_tiempo: pd.DataFrame,
    dim_mpio: pd.DataFrame,
    dim_cb: pd.DataFrame,      # se mantiene en la firma, pero no lo usamos
    dim_victima: pd.DataFrame,
    dim_evento: pd.DataFrame,  # tampoco lo usamos aquí, pero mantenemos la firma
    dim_agresor: pd.DataFrame,
) -> pd.DataFrame:
    """Construye el consolidado mensual con las columnas solicitadas.

    Columnas finales:
    - fecha
    - anio_mes
    - anio_hecho
    - mes_num
    - nombre_municipio
    - estrato
    - sexo_victima
    - sexo_agresor
    - casos
    """

    # 1. Partimos de la fact y unimos SOLO lo necesario
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
            dim_victima[["id_victima", "sexo_", "estrato_"]],
            on="id_victima",
            how="left",
        )
        .merge(
            dim_agresor[["id_agresor", "sexo_agre"]],
            on="id_agresor",
            how="left",
        )
    )

    # 2. Aseguramos tipos y renombramos variables
    ft = ft.dropna(subset=["fecha"])
    ft["anio_hecho"] = ft["anio_hecho"].astype(int)
    ft["mes_num"] = ft["mes_num"].astype(int)

    ft["sexo_victima"] = ft["sexo_"]
    ft["sexo_agresor"] = ft["sexo_agre"]
    ft["estrato"] = ft["estrato_"]

    # Estrato sin dato lo dejamos claro
    ft["estrato"] = ft["estrato"].fillna("Sin Dato")

    # 3. Agrupamos SOLO por los campos que quieres usar como filtros
    group_cols = [
        "fecha",
        "anio_hecho",
        "mes_num",
        "nombre_municipio",
        "estrato",
        "sexo_victima",
        "sexo_agresor",
    ]
    group_cols = [c for c in group_cols if c in ft.columns]

    serie = ft.groupby(group_cols, as_index=False)["casos"].sum()

    # 4. Columna auxiliar YYYY-MM
    serie["anio_mes"] = (
        serie["anio_hecho"].astype(str)
        + "-"
        + serie["mes_num"].astype(str).str.zfill(2)
    )

    # 5. Orden final de columnas
    serie = serie[
        [
            "fecha",
            "anio_mes",
            "anio_hecho",
            "mes_num",
            "nombre_municipio",
            "estrato",
            "sexo_victima",
            "sexo_agresor",
            "casos",
        ]
    ].sort_values(["nombre_municipio", "fecha", "sexo_victima", "sexo_agresor"])

    return serie




# ----------------------------------------------------------------------
# Exportar modelo a Excel y series a CSV
# ----------------------------------------------------------------------
def export_to_excel_model(
    path: Path,
    dim_tiempo: pd.DataFrame,
    dim_depto: pd.DataFrame,
    dim_mpio: pd.DataFrame,
    dim_cb: pd.DataFrame,
    dim_victima: pd.DataFrame,
    dim_evento: pd.DataFrame,
    dim_agresor: pd.DataFrame,
    dim_acciones: pd.DataFrame,
    fact_hechos: pd.DataFrame,
    serie_mensual: pd.DataFrame,
) -> None:
    """Escribe todas las tablas del modelo en un solo archivo Excel."""
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        dim_tiempo.to_excel(writer, sheet_name="DimTiempo", index=False)
        dim_depto.to_excel(writer, sheet_name="DimDepartamento", index=False)
        dim_mpio.to_excel(writer, sheet_name="DimMunicipio", index=False)
        if dim_cb is not None and not dim_cb.empty:
            dim_cb.to_excel(writer, sheet_name="DimComunaBarrio", index=False)
        dim_victima.to_excel(writer, sheet_name="DimVictima", index=False)
        dim_evento.to_excel(writer, sheet_name="DimEvento", index=False)
        dim_agresor.to_excel(writer, sheet_name="DimAgresor", index=False)
        dim_acciones.to_excel(writer, sheet_name="DimAcciones", index=False)
        fact_hechos.to_excel(writer, sheet_name="FactHechos", index=False)
        serie_mensual.to_excel(writer, sheet_name="SerieMensual", index=False)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main() -> None:
    """Ejecuta el ETL completo para Antioquia."""
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    dane_path = DATA_RAW / "municipios_dane.xlsx"
    siv_path = DATA_RAW / "sivigila_violencia.csv"

    df_dane = load_dane_municipios(dane_path)
    df_siv = load_sivigila(siv_path)

    # 1. Filtrar Antioquia y mapear municipios
    df_ant = filter_antioquia_and_map_municipios(df_siv, df_dane)

    # 2. Dimensiones
    dim_tiempo = build_dim_tiempo(df_ant)
    dim_depto = build_dim_departamento(df_ant)
    dim_mpio = build_dim_municipio(df_ant)
    dim_cb = build_dim_comuna_barrio(df_ant)
    dim_victima = build_dim_victima(df_ant)
    dim_evento = build_dim_evento(df_ant)
    dim_agresor = build_dim_agresor(df_ant)
    dim_acciones = build_dim_acciones(df_ant)

    # 3. FactHechos
    fact_hechos = build_fact_hechos(
        df_ant,
        dim_tiempo,
        dim_mpio,
        dim_cb,
        dim_victima,
        dim_evento,
        dim_agresor,
        dim_acciones,
    )

    # 4. Serie mensual por municipio
    serie_mensual = build_serie_mensual_consolidado(
        fact_hechos,
        dim_tiempo,
        dim_mpio,
        dim_cb,
        dim_victima,
        dim_evento,
        dim_agresor,
    )

    # 5. Exportar a Excel maestro + CSV de serie
    excel_path = DATA_PROCESSED / "SIVIGILA_Violencias_Modelo.xlsx"
    export_to_excel_model(
        excel_path,
        dim_tiempo,
        dim_depto,
        dim_mpio,
        dim_cb,
        dim_victima,
        dim_evento,
        dim_agresor,
        dim_acciones,
        fact_hechos,
        serie_mensual,
    )

    serie_mensual.to_csv(DATA_PROCESSED / "SerieMensual.csv", index=False)

    print(f"Modelo exportado a: {excel_path}")
    print("Serie mensual guardada en data/processed/SerieMensual.csv")


if __name__ == "__main__":
    main()
