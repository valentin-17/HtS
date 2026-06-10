"""Datenbank-Setup (SQLite via SQLAlchemy) und Erst-Seed."""

from __future__ import annotations

import logging
from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from . import config
from .models import Base, DayFeature, Sale


logger = logging.getLogger(__name__)

_engine = create_engine(config.DB_URL, future=True)
_SessionFactory = sessionmaker(bind=_engine, future=True, expire_on_commit=False)


@contextmanager
def session_scope():
    """Transaktionaler Session-Kontext mit Commit/Rollback."""
    session = _SessionFactory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def init_db() -> None:
    """Legt die Tabellen an, falls sie noch nicht existieren."""
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    Base.metadata.create_all(_engine)


def seed_if_empty() -> None:
    """Befüllt Kalender und (beim ersten Start) Verkaufsdaten automatisch."""
    # Lokale Importe vermeiden Zirkularität (calendar_build/ingest nutzen db).
    from .calendar_build import rebuild_calendar
    from .ingest import ingest_file

    with session_scope() as session:
        has_calendar = session.query(DayFeature).first() is not None
        has_sales = session.query(Sale).first() is not None

    if not has_calendar:
        count = rebuild_calendar()
        logger.info("Kalender aufgebaut: %s Tage.", count)

    if not has_sales:
        seed_file = next((p for p in config.SEED_FILE_CANDIDATES if p.exists()), None)
        if seed_file is None:
            logger.warning(
                "Keine Seed-Datei gefunden (%s). DB bleibt leer – bitte über Upload befüllen.",
                ", ".join(str(p.name) for p in config.SEED_FILE_CANDIDATES),
            )
            return
        try:
            stats = ingest_file(seed_file)
            logger.info("Seed aus %s: %s", seed_file.name, stats)
        except Exception as exc:  # noqa: BLE001 - Seed darf den Start nicht verhindern
            logger.error("Seed fehlgeschlagen (%s): %s", seed_file.name, exc)

        # Wetter automatisch abrufen (best effort, blockiert den Start nicht hart).
        try:
            from .weather import refresh_weather

            result = refresh_weather()
            logger.info("Wetterabruf beim Seed: %s", result)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Wetterabruf beim Seed übersprungen: %s", exc)
