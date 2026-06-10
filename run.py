"""Lokaler Startpunkt der Mensa-Webanwendung.

Aufruf:  python run.py
Startet einen lokalen Flask-Server und öffnet den Browser.
"""

from __future__ import annotations

import logging
import os
import threading
import webbrowser

from hts.web import create_app


# 0.0.0.0 = auf allen Netzwerk-Schnittstellen lauschen, damit z. B. das Handy
# im selben WLAN über die LAN-IP des PCs zugreifen kann.
BIND_HOST = "0.0.0.0"
PORT = 5000


def _open_browser() -> None:
    # Lokal im Browser immer über localhost öffnen (0.0.0.0 ist keine aufrufbare Adresse).
    webbrowser.open(f"http://127.0.0.1:{PORT}/")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    app = create_app()

    # Browser nur im Hauptprozess öffnen (nicht im Reloader-Kindprozess).
    if not os.environ.get("WERKZEUG_RUN_MAIN"):
        threading.Timer(1.2, _open_browser).start()

    # debug=False, da der Server im Netzwerk erreichbar ist (der interaktive
    # Werkzeug-Debugger erlaubt sonst Code-Ausführung über das Netz).
    app.run(host=BIND_HOST, port=PORT, debug=False, use_reloader=False)


if __name__ == "__main__":
    main()
