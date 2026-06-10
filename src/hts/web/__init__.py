"""Flask-Webanwendung für die Nachfrage-/Verkaufsdaten der Mensa Uni Trier.

Erzeugt die App über die Factory :func:`create_app`. Lokales Hosting via ``run.py``.
"""

from __future__ import annotations

import logging

from flask import Flask

from . import config
from .db import init_db, seed_if_empty
from .routes import bp


logger = logging.getLogger(__name__)


def create_app() -> Flask:
    """Erzeugt und konfiguriert die Flask-App (App-Factory)."""
    app = Flask(__name__)
    app.config["SECRET_KEY"] = "mensa-trier-local-dev"
    app.config["MAX_CONTENT_LENGTH"] = 64 * 1024 * 1024  # 64 MB Upload-Limit

    # Datenbank anlegen und beim ersten Start aus der Excel-Datei seeden.
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    init_db()
    seed_if_empty()

    app.register_blueprint(bp)
    return app
