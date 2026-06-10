"""Einlesen der Produktions-/Ausgabemengen in die Datenbank.

Format (wie ``data_prep.ipynb``):
    RecId | Prod.Datum | Prod.Art | Rezeptur | Ist-Menge

Robuste Variante: Kopfzeile wird anhand der Spaltennamen gesucht (Excel-Exporte
haben teils eine Titelzeile darüber). Anders als im Notebook werden die
``Oliva``-Kategorien **nicht** verworfen, da sie in der Anzeige verlangt sind.
Doppelte Einträge je (Datum, Prod.Art, Rezeptur) werden per Maximum aggregiert.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd

from . import config
from .calendar_build import is_excluded_day, rebuild_calendar
from .db import session_scope
from .models import Sale


_KNOWN_CATS = {config.normalize_category(c) for c in config.MEAL_CATEGORY_ORDER}


_DATE_FORMATS = (
    "%d.%m.%Y",
    "%Y-%m-%d",
    "%Y-%m-%d %H:%M:%S",
    "%d.%m.%Y %H:%M:%S",
    "%d.%m.%y",
)

_HEADER_ALIASES = {
    "prod.datum": "date",
    "prod datum": "date",
    "datum": "date",
    "prod.art": "prod_art",
    "prod art": "prod_art",
    "rezeptur": "rezeptur",
    "ist-menge": "ist_menge",
    "ist menge": "ist_menge",
    "istmenge": "ist_menge",
}


def _norm(value) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split()).casefold()


def _read_raw(path: Path) -> pd.DataFrame:
    """Liest die Datei komplett als Strings ohne Header-Annahme."""
    suffix = path.suffix.lower()
    if suffix in (".xlsx", ".xls"):
        return pd.read_excel(path, header=None, dtype=str)

    # CSV: deutsche Exporte sind i. d. R. ';'-getrennt und cp1252/latin-1 kodiert.
    for encoding in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            return pd.read_csv(
                path,
                header=None,
                dtype=str,
                sep=";",
                engine="python",
                encoding=encoding,
            )
        except (UnicodeDecodeError, UnicodeError):
            continue
    raise ValueError(f"Datei konnte nicht dekodiert werden: {path}")


def _find_header(raw: pd.DataFrame) -> tuple[int, dict[str, int]]:
    """Findet die Kopfzeile und ordnet Spaltenindizes den Zielfeldern zu."""
    for row_idx in range(min(len(raw), 10)):
        cells = {col: _norm(raw.iat[row_idx, col]) for col in range(raw.shape[1])}
        mapping: dict[str, int] = {}
        for col, text in cells.items():
            target = _HEADER_ALIASES.get(text)
            if target and target not in mapping:
                mapping[target] = col
        if {"date", "prod_art"}.issubset(mapping):
            return row_idx, mapping
    raise ValueError(
        "Kopfzeile mit 'Prod.Datum' und 'Prod.Art' nicht gefunden – Format prüfen."
    )


def _parse_date(value) -> dt.date | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, (dt.datetime, pd.Timestamp)):
        return value.date()
    if isinstance(value, dt.date):
        return value
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    for fmt in _DATE_FORMATS:
        try:
            return dt.datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    # Letzter Versuch: pandas-Parser (dayfirst für deutsches Format).
    parsed = pd.to_datetime(text, dayfirst=True, errors="coerce")
    return None if pd.isna(parsed) else parsed.date()


def _parse_menge(value) -> int | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    text = text.replace(".", "").replace(",", ".") if text.count(",") else text
    try:
        return int(round(float(text)))
    except ValueError:
        return None


def parse_file(path: Path) -> list[dict]:
    """Parst eine Datei zu deduplizierten Verkaufszeilen (max je Schlüssel)."""
    raw = _read_raw(path)
    header_idx, mapping = _find_header(raw)

    rez_col = mapping.get("rezeptur")
    menge_col = mapping.get("ist_menge")

    aggregated: dict[tuple, int | None] = {}
    for row_idx in range(header_idx + 1, len(raw)):
        day = _parse_date(raw.iat[row_idx, mapping["date"]])
        if day is None:
            continue
        prod_art = str(raw.iat[row_idx, mapping["prod_art"]] or "").strip()
        if not prod_art or prod_art.lower() == "nan":
            continue
        rezeptur = ""
        if rez_col is not None:
            rezeptur = str(raw.iat[row_idx, rez_col] or "").strip()
            if rezeptur.lower() == "nan":
                rezeptur = ""
        menge = _parse_menge(raw.iat[row_idx, menge_col]) if menge_col is not None else None

        key = (day, prod_art, rezeptur)
        prev = aggregated.get(key, ...)
        if prev is ... or prev is None or (menge is not None and menge > prev):
            aggregated[key] = menge

    return [
        {"date": k[0], "prod_art": k[1], "rezeptur": k[2], "ist_menge": v}
        for k, v in aggregated.items()
    ]


def ingest_file(path: str | Path) -> dict:
    """Liest eine Datei ein, upsertet in die DB und aktualisiert den Kalender.

    Gibt eine kleine Statistik zurück (eingefügt/aktualisiert/Zeilen gesamt).
    """
    path = Path(path)
    rows = parse_file(path)

    # Wochenenden und Feiertage werden nicht importiert (Mensa hat dann geschlossen).
    skipped_excluded = sum(1 for r in rows if is_excluded_day(r["date"]))
    rows = [r for r in rows if not is_excluded_day(r["date"])]

    corrupted = sum(
        1 for r in rows if "�" in (r["prod_art"] or "") or "�" in (r["rezeptur"] or "")
    )
    unknown_categories = sum(
        1 for r in rows if config.normalize_category(r["prod_art"]) not in _KNOWN_CATS
    )

    inserted = 0
    updated = 0
    with session_scope() as session:
        existing = {
            (s.date, s.prod_art, s.rezeptur): s
            for s in session.query(Sale).all()
        }
        for row in rows:
            key = (row["date"], row["prod_art"], row["rezeptur"])
            sale = existing.get(key)
            if sale is None:
                session.add(Sale(**row))
                inserted += 1
            else:
                if sale.ist_menge != row["ist_menge"]:
                    sale.ist_menge = row["ist_menge"]
                    updated += 1

    # Kalender für (ggf. neue) Tage sicherstellen.
    rebuild_calendar()

    return {
        "rows": len(rows),
        "inserted": inserted,
        "updated": updated,
        "corrupted": corrupted,
        "unknown_categories": unknown_categories,
        "skipped_excluded": skipped_excluded,
    }


def purge_excluded_sales() -> int:
    """Löscht alle vorhandenen Verkaufszeilen an Wochenenden und Feiertagen.

    Einmalige Bereinigung bereits eingelesener Daten; gibt die Anzahl gelöschter Zeilen zurück.
    """
    removed = 0
    with session_scope() as session:
        for sale in session.query(Sale).all():
            if is_excluded_day(sale.date):
                session.delete(sale)
                removed += 1
    return removed
