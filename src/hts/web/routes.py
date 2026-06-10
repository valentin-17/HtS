"""Flask-Routen: Kalender (Monat/Woche/Tag), Upload, Wetter, Datenanalyse."""

from __future__ import annotations

import calendar as _cal
import datetime as dt
import tempfile
from pathlib import Path

from flask import (
    Blueprint,
    abort,
    flash,
    redirect,
    render_template,
    request,
    url_for,
)
from werkzeug.utils import secure_filename

from . import config
from .calendar_build import is_excluded_day
from .db import session_scope
from .models import DayFeature, Sale


bp = Blueprint("main", __name__)

_CATEGORY_BY_NORM = {config.normalize_category(c): c for c in config.MEAL_CATEGORY_ORDER}


@bp.app_context_processor
def inject_today():
    """Stellt das heutige Datum (Systemuhr) allen Templates bereit."""
    today = dt.date.today()
    iso = today.isocalendar()
    return {"current_day": today, "current_week": (iso.year, iso.week)}


# --- Datenzugriff --------------------------------------------------------
def _sales_in_range(start: dt.date, end: dt.date) -> dict[dt.date, list[dict]]:
    """Liefert je Tag eine Liste von {prod_art, rezeptur, ist_menge}."""
    result: dict[dt.date, list[dict]] = {}
    with session_scope() as session:
        rows = (
            session.query(Sale)
            .filter(Sale.date >= start, Sale.date <= end)
            .order_by(Sale.date, Sale.prod_art, Sale.rezeptur)
            .all()
        )
        for s in rows:
            result.setdefault(s.date, []).append(
                {"prod_art": s.prod_art, "rezeptur": s.rezeptur, "ist_menge": s.ist_menge}
            )
    return result


def _features_in_range(start: dt.date, end: dt.date) -> dict[dt.date, dict]:
    out: dict[dt.date, dict] = {}
    with session_scope() as session:
        rows = (
            session.query(DayFeature)
            .filter(DayFeature.date >= start, DayFeature.date <= end)
            .all()
        )
        for f in rows:
            out[f.date] = {
                "weekday": f.weekday,
                "kw": f.kw,
                "is_weekend": f.is_weekend,
                "academic_bucket": f.academic_bucket,
                "is_public_holiday": f.is_public_holiday,
                "public_holiday_name": f.public_holiday_name,
                "is_university_closure": f.is_university_closure,
                "semester": f.semester,
                "sunshine_duration": f.sunshine_duration,
                "precipitation_hours": f.precipitation_hours,
                "apparent_temperature_mean": f.apparent_temperature_mean,
                "precipitation_sum": f.precipitation_sum,
                "weather_category": f.weather_category,
            }
    return out


def _preview_menus(day_sales: list[dict]) -> list[dict]:
    """Menü 1 & 2 mit summierter Ist-Menge (für die Monatsvorschau)."""
    preview = []
    for category in config.MONTH_PREVIEW_CATEGORIES:
        norm = config.normalize_category(category)
        matches = [s for s in day_sales if config.normalize_category(s["prod_art"]) == norm]
        if matches:
            mengen = [s["ist_menge"] for s in matches if s["ist_menge"] is not None]
            total = sum(mengen) if mengen else None
        else:
            total = None
        preview.append({"category": category, "ist_menge": total})
    return preview


def _unknown_entries(day_sales: list[dict]) -> list[dict]:
    """Zeilen, deren Prod.Art zu keiner der 16 Kategorien passt (z. B. kaputte Kodierung)."""
    return [s for s in day_sales if config.normalize_category(s["prod_art"]) not in _CATEGORY_BY_NORM]


def _menus_with_recipe(day_sales: list[dict]) -> list[dict]:
    """Menü 1 & 2 mit Rezeptur + Ist-Menge (für die Wochenansicht)."""
    items = []
    for category in config.MONTH_PREVIEW_CATEGORIES:
        norm = config.normalize_category(category)
        matches = [s for s in day_sales if config.normalize_category(s["prod_art"]) == norm]
        items.append({"category": category, "entries": matches})
    unknown = _unknown_entries(day_sales)
    if unknown:
        items.append({"category": config.UNKNOWN_CATEGORY_LABEL, "entries": unknown, "unknown": True})
    return items


def _full_day(day_sales: list[dict]) -> list[dict]:
    """Alle 16 Kategorien in fester Reihenfolge mit (Rezeptur, Ist-Menge) oder N/A."""
    full = []
    for category in config.MEAL_CATEGORY_ORDER:
        norm = config.normalize_category(category)
        matches = [s for s in day_sales if config.normalize_category(s["prod_art"]) == norm]
        full.append({"category": category, "entries": matches})
    unknown = _unknown_entries(day_sales)
    if unknown:
        full.append({"category": config.UNKNOWN_CATEGORY_LABEL, "entries": unknown, "unknown": True})
    return full


def _shift_month(year: int, month: int, delta_months: int) -> tuple[int, int]:
    index = (year * 12 + (month - 1)) + delta_months
    return index // 12, index % 12 + 1


# --- Routen --------------------------------------------------------------
@bp.route("/")
def index():
    ref = config.REFERENCE_DATE
    return redirect(url_for("main.month_view", year=ref.year, month=ref.month))


@bp.route("/kalender/monat/<int:year>/<int:month>")
def month_view(year: int, month: int):
    if not 1 <= month <= 12:
        abort(404)

    weeks_dates = _cal.Calendar(firstweekday=0).monthdatescalendar(year, month)
    start = weeks_dates[0][0]
    end = weeks_dates[-1][-1]

    sales = _sales_in_range(start, end)
    features = _features_in_range(start, end)

    weeks = []
    for week in weeks_dates:
        kw = week[3].isocalendar().week  # ISO-KW über den Donnerstag
        days = []
        for day in week:
            day_sales = sales.get(day, [])
            feat = features.get(day, {})
            days.append(
                {
                    "date": day,
                    "in_month": day.month == month,
                    "feature": feat,
                    "preview": _preview_menus(day_sales),
                    "has_sales": bool(day_sales),
                    "excluded": is_excluded_day(day),
                }
            )
        weeks.append({"kw": kw, "iso_year": week[3].isocalendar().year, "days": days})

    prev_y, prev_m = _shift_month(year, month, -1)
    next_y, next_m = _shift_month(year, month, 1)
    half_y, half_m = _shift_month(year, month, -6)
    year_y, year_m = _shift_month(year, month, -12)

    return render_template(
        "month.html",
        year=year,
        month=month,
        month_name=_german_month(month),
        weeks=weeks,
        weekday_headers=["Mo", "Di", "Mi", "Do", "Fr", "Sa", "So"],
        anchor=dt.date(year, month, 1),
        nav={
            "prev": (prev_y, prev_m),
            "next": (next_y, next_m),
            "half_year": (half_y, half_m),
            "one_year": (year_y, year_m),
        },
    )


@bp.route("/kalender/woche/<int:year>/<int:week>")
def week_view(year: int, week: int):
    try:
        monday = dt.date.fromisocalendar(year, week, 1)
    except ValueError:
        abort(404)

    days_dates = [monday + dt.timedelta(days=i) for i in range(5)]  # Mo–Fr
    sales = _sales_in_range(days_dates[0], days_dates[-1])
    features = _features_in_range(days_dates[0], days_dates[-1])

    days = []
    for day in days_dates:
        day_sales = sales.get(day, [])
        days.append(
            {
                "date": day,
                "feature": features.get(day, {}),
                "menus": _menus_with_recipe(day_sales),
                "excluded": is_excluded_day(day),
            }
        )

    from .forecast_mock import MOCKUP_WEEK, MOCKUP_YEAR, forecast_mockup_data_uri

    return render_template(
        "week.html",
        year=year,
        week=week,
        days=days,
        forecast_img_menu1=forecast_mockup_data_uri(MOCKUP_YEAR, MOCKUP_WEEK, "Menü 1"),
        forecast_img_menu2=forecast_mockup_data_uri(MOCKUP_YEAR, MOCKUP_WEEK, "Menü 2"),
        anchor=monday,
        prev_week=_shift_week(year, week, -1),
        next_week=_shift_week(year, week, 1),
        month_of_week=(monday.year, monday.month),
    )


@bp.route("/kalender/tag/<day>")
def day_view(day: str):
    try:
        the_day = dt.date.fromisoformat(day)
    except ValueError:
        abort(404)

    sales = _sales_in_range(the_day, the_day).get(the_day, [])
    feature = _features_in_range(the_day, the_day).get(the_day, {})
    iso = the_day.isocalendar()

    return render_template(
        "day.html",
        the_day=the_day,
        anchor=the_day,
        feature=feature,
        excluded=is_excluded_day(the_day),
        categories=_full_day(sales),
        week=(iso.year, iso.week),
        month=(the_day.year, the_day.month),
        prev_day=the_day - dt.timedelta(days=1),
        next_day=the_day + dt.timedelta(days=1),
    )


@bp.route("/upload", methods=["GET", "POST"])
def upload():
    if request.method == "POST":
        file = request.files.get("file")
        if not file or not file.filename:
            flash("Keine Datei ausgewählt.", "error")
            return redirect(url_for("main.upload"))

        filename = secure_filename(file.filename)
        suffix = Path(filename).suffix.lower()
        if suffix not in config.UPLOAD_EXTENSIONS:
            flash(f"Dateiformat {suffix} nicht unterstützt (erlaubt: xlsx, xls, csv).", "error")
            return redirect(url_for("main.upload"))

        from .ingest import ingest_file

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            file.save(tmp.name)
            tmp_path = Path(tmp.name)
        try:
            stats = ingest_file(tmp_path)
            flash(
                f"Import erfolgreich: {stats['rows']} Zeilen verarbeitet, "
                f"{stats['inserted']} neu, {stats['updated']} aktualisiert.",
                "success",
            )
            if stats.get("corrupted"):
                flash(
                    f"⚠ {stats['corrupted']} Zeile(n) enthalten beschädigte Zeichen (�). "
                    "Datei vermutlich falsch kodiert – bitte als CSV (Windows-1252) aus Excel "
                    "exportieren.",
                    "warning",
                )
            if stats.get("unknown_categories"):
                flash(
                    f"⚠ {stats['unknown_categories']} Zeile(n) mit unbekannter Prod.Art "
                    "(nicht unter den 16 Kategorien) – sie erscheinen unter „Sonstige / Unbekannt“.",
                    "warning",
                )
            if stats.get("skipped_excluded"):
                flash(
                    f"ℹ {stats['skipped_excluded']} Zeile(n) an Wochenenden/Feiertagen wurden "
                    "ignoriert – an diesen Tagen gibt es keine Mensa-Verkäufe.",
                    "warning",
                )
        except Exception as exc:  # noqa: BLE001
            flash(f"Import fehlgeschlagen: {exc}", "error")
        finally:
            tmp_path.unlink(missing_ok=True)
        return redirect(url_for("main.upload"))

    return render_template("upload.html")


@bp.route("/wetter/aktualisieren", methods=["POST"])
def refresh_weather_route():
    from .weather import refresh_weather

    try:
        result = refresh_weather()
        if result.get("error"):
            flash(f"Wetterabruf: {result['error']}", "error")
        else:
            flash(f"Wetterdaten aktualisiert ({result['updated']} Tage).", "success")
    except Exception as exc:  # noqa: BLE001
        flash(f"Wetterabruf fehlgeschlagen: {exc}", "error")

    ref = request.form.get("return_to") or url_for("main.index")
    return redirect(ref)


@bp.route("/auswertung")
def analyse():
    from .plots import build_plots

    plots = build_plots()
    return render_template("analyse.html", plots=plots)


# --- Hilfen --------------------------------------------------------------
_GERMAN_MONTHS = [
    "Januar", "Februar", "März", "April", "Mai", "Juni",
    "Juli", "August", "September", "Oktober", "November", "Dezember",
]


def _german_month(month: int) -> str:
    return _GERMAN_MONTHS[month - 1]


def _shift_week(year: int, week: int, delta: int) -> tuple[int, int]:
    monday = dt.date.fromisocalendar(year, week, 1) + dt.timedelta(weeks=delta)
    iso = monday.isocalendar()
    return iso.year, iso.week


@bp.app_template_filter("menge")
def _format_menge(value):
    """Formatiert eine Ist-Menge oder zeigt N/A."""
    if value is None:
        return "N/A"
    return value
