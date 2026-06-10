"""Kalender- und Feiertagslogik für die Uni/Mensa Trier.

Portierung der Logik aus ``src/hts/data_prep.ipynb`` von einem festen Jahr (2025)
auf einen konfigurierbaren Jahresbereich. Erzeugt/aktualisiert die DayFeature-Zeilen.
"""

from __future__ import annotations

from datetime import date, timedelta
from functools import lru_cache

import holidays

from . import config
from .db import session_scope
from .models import DayFeature


WEEKDAY_DE = {
    0: "Montag",
    1: "Dienstag",
    2: "Mittwoch",
    3: "Donnerstag",
    4: "Freitag",
    5: "Samstag",
    6: "Sonntag",
}


@lru_cache(maxsize=None)
def _holidays_for_year(year: int):
    """Gesetzliche Feiertage in Rheinland-Pfalz für ein Jahr (gecacht)."""
    return holidays.country_holidays("DE", subdiv="RP", years=year)


def is_public_holiday(day: date) -> bool:
    """True, wenn ``day`` ein gesetzlicher Feiertag (RP) ist."""
    return day in _holidays_for_year(day.year)


def is_excluded_day(day: date) -> bool:
    """Wochenende (Sa/So) oder gesetzlicher Feiertag – an diesen Tagen gibt es keine Mensa-Verkäufe.

    Solche Tage werden weder importiert noch angezeigt.
    """
    return day.weekday() >= 5 or is_public_holiday(day)


def _semester_for(day: date) -> dict | None:
    for term in config.SEMESTER_TERMS:
        if term["semester_start"] <= day <= term["semester_end"]:
            return term
    return None


def _is_in_lecture(day: date) -> bool:
    for term in config.SEMESTER_TERMS:
        if term["lecture_start"] <= day <= term["lecture_end"]:
            return True
    return False


def _is_university_holiday(day: date) -> bool:
    for start, end in config.UNIVERSITY_HOLIDAYS.values():
        if start <= day <= end:
            return True
    return False


def daterange(start: date, end: date):
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)


def build_calendar_rows() -> list[dict]:
    """Berechnet die Tages-Features für den konfigurierten Jahresbereich."""
    years = list(range(config.CALENDAR_START_YEAR, config.CALENDAR_END_YEAR + 1))
    public_holidays = holidays.country_holidays("DE", subdiv="RP", years=years)

    start = date(config.CALENDAR_START_YEAR, 1, 1)
    end = date(config.CALENDAR_END_YEAR, 12, 31)

    rows: list[dict] = []
    for day in daterange(start, end):
        semester = _semester_for(day)
        is_weekend = day.weekday() >= 5
        is_public = day in public_holidays
        is_uni_holiday = _is_university_holiday(day)
        is_closure = is_public or is_uni_holiday

        if is_closure:
            bucket = "university_closure"
        elif _is_in_lecture(day):
            bucket = "lecture"
        elif semester is not None:
            bucket = "lecture_free"
        else:
            bucket = "semester_break"

        rows.append(
            {
                "date": day,
                "weekday": WEEKDAY_DE[day.weekday()],
                "kw": day.isocalendar().week,
                "is_weekend": is_weekend,
                "semester": semester["semester"] if semester else None,
                "academic_bucket": bucket,
                "is_public_holiday": is_public,
                "public_holiday_name": public_holidays.get(day),
                "is_university_closure": is_closure,
            }
        )
    return rows


def rebuild_calendar() -> int:
    """Schreibt/aktualisiert alle DayFeature-Kalenderfelder. Wetter bleibt erhalten."""
    rows = build_calendar_rows()
    with session_scope() as session:
        existing = {df.date: df for df in session.query(DayFeature).all()}
        for row in rows:
            df = existing.get(row["date"])
            if df is None:
                session.add(DayFeature(**row))
            else:
                for key, value in row.items():
                    setattr(df, key, value)
    return len(rows)
