"""Mock-up-Grafik einer Wochenprognose (Mo–Fr) für die Wochenansicht.

Rein illustrativ: zeigt, wie eine Prognose mit Konfidenzintervall (gestrichelte
Upper-/Lower-Linien), Prognoselinie und 10-Tage-Durchschnitt aussehen könnte.
Die Werte sind synthetisch, aber pro Kalenderwoche deterministisch (gleiche
Woche → gleiches Bild). Rückgabe als ``data:``-URI (PNG, base64), damit sie ohne
Datei-Handling direkt in ein ``<img>`` eingebettet werden kann.

Es wird bewusst die objektorientierte Matplotlib-API (``Figure``) genutzt – ohne
globalen ``pyplot``-Zustand, damit das Rendern im Flask-Server thread-sicher ist.
"""

from __future__ import annotations

import base64
import random
from functools import lru_cache
from io import BytesIO

from matplotlib.figure import Figure


WEEKDAYS = ["Mo", "Di", "Mi", "Do", "Fr"]

# Feste Beispielwoche: die Prognose-Vorschau zeigt in jeder Wochenansicht
# dasselbe Mock-up (KW 25), unabhängig von der tatsächlich angezeigten Woche.
MOCKUP_YEAR = 2026
MOCKUP_WEEK = 25

# Zwei Menüs als getrennte Mock-ups. Menü 1 wird im Schnitt ~100 Portionen mehr
# verkauft als Menü 2 (über das Grundniveau ``base`` modelliert). ``seed_offset``
# sorgt für unterschiedliche, aber je Woche stabile Tagesverläufe.
MENU_SPEC = {
    "Menü 1": {"base": 230.0, "seed_offset": 1},
    "Menü 2": {"base": 130.0, "seed_offset": 2},
}

# Farben passend zur studiwerk.css-Palette.
_GREEN = "#53A12E"      # Prognose
_YELLOW = "#F3B700"     # 10-Tage-Durchschnitt
_BAND = "#85BAE2"       # Konfidenzband (Füllung)
_GRAY = "#9E9E9E"       # gestrichelte Intervallgrenzen


@lru_cache(maxsize=256)
def forecast_mockup_data_uri(year: int, week: int, menu: str = "Menü 1") -> str:
    """Erzeugt das Prognose-Mockup eines Menüs als ``data:``-URI (PNG).

    Gecacht je (Woche, Menü); deterministisch über die Server-Laufzeit hinweg.
    """
    spec = MENU_SPEC.get(menu, {"base": 180.0, "seed_offset": 0})
    rng = random.Random(year * 1000 + week * 10 + spec["seed_offset"])

    base = spec["base"]
    forecast = [max(0.0, base + rng.uniform(-45, 45)) for _ in WEEKDAYS]
    mean = sum(forecast) / len(forecast)
    ten_day_avg = [mean + rng.uniform(-12, 12) for _ in WEEKDAYS]
    margin = [rng.uniform(35, 60) for _ in WEEKDAYS]
    upper = [f + m for f, m in zip(forecast, margin)]
    lower = [max(0.0, f - m) for f, m in zip(forecast, margin)]

    fig = Figure(figsize=(7.5, 3.4), dpi=110)
    ax = fig.subplots()
    x = list(range(len(WEEKDAYS)))

    ax.fill_between(x, lower, upper, color=_BAND, alpha=0.18)
    ax.plot(x, upper, linestyle="--", color=_GRAY, linewidth=1.4, label="Oberes Intervall (Upper)")
    ax.plot(x, lower, linestyle="--", color=_GRAY, linewidth=1.4, label="Unteres Intervall (Lower)")
    ax.plot(x, ten_day_avg, color=_YELLOW, linewidth=2.0, label="10-Tage-Durchschnitt")
    ax.plot(x, forecast, color=_GREEN, linewidth=2.2, marker="o", label="Prognose")

    ax.set_title(f"Prognose-Mockup {menu} – KW {week} / {year} (Beispieldaten)")
    ax.set_xlabel("Wochentag")
    ax.set_ylabel("Verkäufe")
    ax.set_xticks(x)
    ax.set_xticklabels(WEEKDAYS)
    ax.set_ylim(bottom=0)
    ax.grid(True, axis="y", linestyle=":", alpha=0.4)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=4, fontsize=8, frameon=False)

    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"
