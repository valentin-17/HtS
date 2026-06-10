"""Zentrale Konfiguration der Mensa-Webanwendung."""

from __future__ import annotations

from datetime import date
from pathlib import Path


# --- Pfade ---------------------------------------------------------------
# Projekt-Root: .../HtS  (drei Ebenen über dieser Datei: web -> hts -> src -> HtS)
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data"
DB_PATH = DATA_DIR / "mensa.db"
DB_URL = f"sqlite:///{DB_PATH}"

# Quelle für den automatischen Erst-Seed. Es wird die erste existierende Datei
# verwendet (CSV bevorzugt, xlsx nur als Fallback).
SEED_FILE_CANDIDATES = [
    DATA_DIR / "private" / "Produktions-Ausgabemengen Mensa Tarforst_Mensa Oliva.csv",
    DATA_DIR / "private" / "Produktions-Ausgabemengen Mensa Tarforst_Mensa Oliva.xlsx",
]

UPLOAD_EXTENSIONS = {".xlsx", ".xls", ".csv"}

# --- Standort / Wetter ---------------------------------------------------
# Trier (wie in data_prep.ipynb)
TRIER_LATITUDE = 49.7557
TRIER_LONGITUDE = 6.6394
WEATHER_TIMEZONE = "Europe/Berlin"

# --- Kalender ------------------------------------------------------------
# "Jetzt" laut Use-Case = April 2026 (konfigurierbar). Steuert die Startansicht
# und die Buttons "6 Monate zurück" / "1 Jahr zurück".
REFERENCE_DATE = date(2026, 4, 15)

# Jahresbereich, für den Kalender/Feiertage vorberechnet werden.
CALENDAR_START_YEAR = 2024
CALENDAR_END_YEAR = 2026

# Semester der Uni Trier (portiert aus data_prep.ipynb).
SEMESTER_TERMS = [
    {
        "semester": "winter_2024_2025",
        "semester_season": "winter",
        "semester_start": date(2024, 10, 1),
        "semester_end": date(2025, 3, 31),
        "lecture_start": date(2024, 10, 21),
        "lecture_end": date(2025, 2, 15),
    },
    {
        "semester": "summer_2025",
        "semester_season": "summer",
        "semester_start": date(2025, 4, 1),
        "semester_end": date(2025, 9, 30),
        "lecture_start": date(2025, 4, 14),
        "lecture_end": date(2025, 7, 19),
    },
    {
        "semester": "winter_2025_2026",
        "semester_season": "winter",
        "semester_start": date(2025, 10, 1),
        "semester_end": date(2026, 3, 31),
        "lecture_start": date(2025, 10, 13),
        "lecture_end": date(2026, 2, 14),
    },
    {
        "semester": "summer_2026",
        "semester_season": "summer",
        "semester_start": date(2026, 4, 1),
        "semester_end": date(2026, 9, 30),
        "lecture_start": date(2026, 4, 13),
        "lecture_end": date(2026, 7, 17),
    },
]

# Vorlesungsfreie Uni-Schließzeiten (Weihnachtspausen).
UNIVERSITY_HOLIDAYS = {
    "winter_2024_2025_christmas_break": (date(2024, 12, 23), date(2025, 1, 5)),
    "winter_2025_2026_christmas_break": (date(2025, 12, 22), date(2026, 1, 7)),
}

# --- Anzeige -------------------------------------------------------------
# Feste Reihenfolge der 16 Produkt-Arten (Prod.Art) für die Tagesansicht.
MEAL_CATEGORY_ORDER = [
    "Menü 1",
    "Menü 2",
    "Eintopf groß",
    "Tellergericht Pasta Cor.",
    "Pasta - Oliva",
    "Pizza - Oliva",
    "Bowls - Oliva",
    "Salatteller",
    "Beilagen/Salate",
    "Beilagen/Stärkeprodukte",
    "Beilagen",
    "Toppings - Oliva",
    "Nachschlag in der Mensa",
    "Sweet Joker",
    "Joker",
    "Dessert",
]

# Kategorien für die Monatsvorschau (nur Menü 1 & 2 mit Ist-Menge).
MONTH_PREVIEW_CATEGORIES = ["Menü 1", "Menü 2"]

# Sammelkategorie für Zeilen, deren Prod.Art zu keiner der 16 Kategorien passt
# (z. B. durch falsche Datei-Kodierung beschädigte Werte wie "Men� 2").
UNKNOWN_CATEGORY_LABEL = "Sonstige / Unbekannt"

# Studierendenwerk-Trier-Farbpalette (aus data_prep.ipynb / dish_sales_plot.py).
COLORS = {
    "gruen": "#53A12E",
    "rot": "#D02525",
    "gelb": "#F3B700",
    "dunkelblau": "#22608F",
    "blau": "#3F92D2",
    "hellblau": "#85BAE2",
    "grau": "#9E9E9E",
}

SEGMENT_COLORS = {
    "Konsistent starke Verkäufe": "#53A12E",
    "Konsistent schwache Verkäufe": "#D02525",
    "Starke Verkäufe aber volatil": "#F3B700",
    "Niedrige Verkäufe und unvorhersehbar": "#3F92D2",
    "Mittlerer Bereich": "#9E9E9E",
}


def normalize_category(value: str | None) -> str:
    """Normalisiert eine Prod.Art für case-/leerzeichen-toleranten Vergleich."""
    if value is None:
        return ""
    return " ".join(value.split()).casefold()
