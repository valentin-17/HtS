"""Automatischer Abruf externer Wetterdaten (Open-Meteo) für Trier.

Portierung aus ``data_prep.ipynb``: Archiv-API für die Vergangenheit, Forecast-API
für die kommenden Tage. Ergebnisse landen in den Wetterspalten von ``DayFeature``.
"""

from __future__ import annotations

import datetime as dt
import logging

import openmeteo_requests
import pandas as pd
import requests_cache
from retry_requests import retry

from . import config
from .db import session_scope
from .models import DayFeature


logger = logging.getLogger(__name__)

_DAILY_VARS = [
    "sunshine_duration",
    "precipitation_hours",
    "apparent_temperature_mean",
    "precipitation_sum",
]

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"


def _client() -> openmeteo_requests.Client:
    cache_session = requests_cache.CachedSession(
        str(config.PROJECT_ROOT / ".cache"), expire_after=-1
    )
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    return openmeteo_requests.Client(session=retry_session)


def _fetch(client, url: str, start: dt.date, end: dt.date) -> dict[dt.date, dict]:
    params = {
        "latitude": config.TRIER_LATITUDE,
        "longitude": config.TRIER_LONGITUDE,
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "daily": _DAILY_VARS,
        "timezone": config.WEATHER_TIMEZONE,
    }
    responses = client.weather_api(url, params=params)
    response = responses[0]
    daily = response.Daily()

    dates = pd.date_range(
        start=pd.to_datetime(daily.Time(), unit="s", utc=True),
        end=pd.to_datetime(daily.TimeEnd(), unit="s", utc=True),
        freq=pd.Timedelta(seconds=daily.Interval()),
        inclusive="left",
    ).tz_convert(config.WEATHER_TIMEZONE)

    values = {var: daily.Variables(i).ValuesAsNumpy() for i, var in enumerate(_DAILY_VARS)}

    out: dict[dt.date, dict] = {}
    for idx, ts in enumerate(dates):
        day = ts.date()
        out[day] = {
            # Sekunden -> Stunden (wie im Notebook)
            "sunshine_duration": float(values["sunshine_duration"][idx]) / 3600.0,
            "precipitation_hours": float(values["precipitation_hours"][idx]),
            "apparent_temperature_mean": float(values["apparent_temperature_mean"][idx]),
            "precipitation_sum": float(values["precipitation_sum"][idx]),
        }
    return out


def _weather_category(rec: dict, mean_precip_sum: float) -> str:
    threshold = 6.0
    sunshine = rec["sunshine_duration"]
    precip_hours = rec["precipitation_hours"]
    precip_sum = rec["precipitation_sum"]
    if sunshine >= threshold and precip_hours < threshold:
        return "sunny"
    if (
        precip_hours >= threshold
        and sunshine < threshold
        and precip_sum >= mean_precip_sum
    ):
        return "rainy"
    return "normal"


def refresh_weather() -> dict:
    """Holt Wetterdaten für den Kalenderbereich und schreibt sie in DayFeature."""
    today = dt.date.today()
    start = dt.date(config.CALENDAR_START_YEAR, 1, 1)
    archive_end = min(today, dt.date(config.CALENDAR_END_YEAR, 12, 31))

    client = _client()
    weather: dict[dt.date, dict] = {}

    try:
        weather.update(_fetch(client, ARCHIVE_URL, start, archive_end))
    except Exception as exc:  # noqa: BLE001 - Netzwerk/External, robust loggen
        logger.warning("Archiv-Wetterabruf fehlgeschlagen: %s", exc)

    forecast_end = min(today + dt.timedelta(days=15), dt.date(config.CALENDAR_END_YEAR, 12, 31))
    if forecast_end >= today:
        try:
            weather.update(_fetch(client, FORECAST_URL, today, forecast_end))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Forecast-Wetterabruf fehlgeschlagen: %s", exc)

    if not weather:
        return {"updated": 0, "error": "Keine Wetterdaten abrufbar."}

    precip_values = [r["precipitation_sum"] for r in weather.values()]
    mean_precip_sum = sum(precip_values) / len(precip_values)

    updated = 0
    with session_scope() as session:
        existing = {df.date: df for df in session.query(DayFeature).all()}
        for day, rec in weather.items():
            df = existing.get(day)
            if df is None:
                df = DayFeature(date=day)
                session.add(df)
            df.sunshine_duration = rec["sunshine_duration"]
            df.precipitation_hours = rec["precipitation_hours"]
            df.apparent_temperature_mean = rec["apparent_temperature_mean"]
            df.precipitation_sum = rec["precipitation_sum"]
            df.weather_category = _weather_category(rec, mean_precip_sum)
            updated += 1

    return {"updated": updated}
