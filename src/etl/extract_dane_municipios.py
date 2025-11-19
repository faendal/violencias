from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import pdfplumber
import unicodedata

# Carpetas base del proyecto
DATA_RAW = Path("data/raw")
DATA_PROCESSED = Path("data/processed")


def normalize_text(value: str) -> str:
    """Normaliza nombres quitando tildes y espacios extra.

    Args:
        value: Texto original.

    Returns:
        Texto normalizado en mayúsculas, sin tildes y con espacios simples.
    """
    if value is None:
        return ""
    text = str(value).strip().upper()
    # Quitar tildes
    text = "".join(
        c for c in unicodedata.normalize("NFKD", text)
        if not unicodedata.combining(c)
    )
    # Reemplazar dobles espacios
    while "  " in text:
        text = text.replace("  ", " ")
    return text


def extract_tables_from_pdf(pdf_path: Path) -> pd.DataFrame:
    """Extrae la tabla de códigos DANE desde el PDF.

    El PDF tiene 4 columnas:
    - Código Departamento
    - Nombre Departamento
    - Código Municipio
    - Nombre Municipio

    Args:
        pdf_path: Ruta al PDF de municipios por departamento.

    Returns:
        DataFrame con las columnas:
        ['cod_dpto', 'nom_dpto', 'cod_mpio', 'nom_mpio',
         'cod_mpio_dane', 'nom_dpto_norm', 'nom_mpio_norm'].
    """
    rows = []

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            tables = page.extract_tables()
            if not tables:
                continue

            table = tables[0]
            if not table:
                continue

            # Detectar si la primera fila son encabezados
            first_cell = table[0][0] or ""
            if "CÓDIGO" in first_cell.upper() or "CODIGO" in first_cell.upper():
                start_idx = 1
            else:
                start_idx = 0

            for row in table[start_idx:]:
                if row is None or len(row) < 4:
                    continue
                cod_dpto, nom_dpto, cod_mpio, nom_mpio = row[:4]
                if not cod_dpto or not cod_mpio:
                    continue

                rows.append(
                    {
                        "cod_dpto": str(cod_dpto).strip().zfill(2),
                        "nom_dpto": str(nom_dpto).strip(),
                        "cod_mpio": str(cod_mpio).strip().zfill(3),
                        "nom_mpio": str(nom_mpio).strip(),
                    }
                )

    df = pd.DataFrame(rows).drop_duplicates()

    if df.empty:
        raise ValueError(
            "No se extrajeron filas del PDF. Revisa que el formato de la tabla no haya cambiado."
        )

    # Código DANE completo de municipio (5 dígitos)
    df["cod_mpio_dane"] = df["cod_dpto"] + df["cod_mpio"]

    # Nombres normalizados para cruzar luego con SIVIGILA
    df["nom_dpto_norm"] = df["nom_dpto"].apply(normalize_text)
    df["nom_mpio_norm"] = df["nom_mpio"].apply(normalize_text)

    return df


def find_pdf_path() -> Path:
    """Busca el PDF en ubicaciones típicas del proyecto.

    Orden de búsqueda:
    1. data/raw/Codificación de Municipios por Departamento.pdf
    2. Codificación de Municipios por Departamento.pdf (raíz del proyecto)

    Returns:
        Ruta al archivo PDF.

    Raises:
        FileNotFoundError: Si no se encuentra el archivo en ninguna ubicación.
    """
    candidate_names = [
        "Codificación de Municipios por Departamento.pdf",
        "Codificacion de Municipios por Departamento.pdf",
    ]

    candidates: list[Path] = []
    for name in candidate_names:
        candidates.append(DATA_RAW / name)
        candidates.append(Path(name))

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(
        "No se encontró el archivo 'Codificación de Municipios por Departamento.pdf'. "
        "Ubícalo en la carpeta 'data/raw/' o en la raíz del proyecto."
    )


def main() -> None:
    """Punto de entrada del script.

    Lee el PDF de municipios por departamento y genera un Excel con el
    maestro DANE que usaremos en el ETL de SIVIGILA.
    """
    pdf_path = find_pdf_path()
    print(f"Usando PDF de DANE ubicado en: {pdf_path}")

    df_dane = extract_tables_from_pdf(pdf_path)

    # Asegurar carpetas de salida
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    # Guardar maestro en raw (para ETL) y en processed (por si lo consumes directo)
    output_raw = DATA_RAW / "municipios_dane.xlsx"
    output_processed = DATA_PROCESSED / "municipios_dane.xlsx"

    df_dane.to_excel(output_raw, index=False)
    df_dane.to_excel(output_processed, index=False)

    print(f"Archivo de municipios DANE generado en: {output_raw}")
    print(f"Archivo de municipios DANE generado en: {output_processed}")


if __name__ == "__main__":
    main()
