# Grundannahmen
- Wir nehmen an, dass die Mensa ihre Produktion korrekt planen kann. Dafür benötigt die Mensa von uns eine erwartete Anzahl an Besuchern und idealerweise eine erwartete Anzahl an verkauften Gerichten pro Menüpunkt.
- Ziel ist es die Prognose des Modells erklären zu können. Das Modell soll zusätzlich zur erwarteten Anzahl an Besuchern / an verkauften Gerichten, Gründe ausgeben, wie das Ergebnis erzielt wurde.
- Beliebtheit von Gerichten hat einen Einfluss auf die Nachfrage: Wenn Gäste sehen, dass nächste Woche ein beliebtes Gericht angeboten wird steigt die Nachfrage

# Datenprep
- Joker / Oliva Produktionsdaten nehmen und korellieren:
  - Wetter
  - ÖPNV Situation -> bekommen wir da gute Daten?
  - Vorlesungszeit vs. Vorlesungsfreie Zeit (+ Klausurenphase?)
  - Wochentag Buckets
  - Events -> Konzerte, Tagungen, Schülertage etc. (siehe https://www.uni-trier.de/universitaet/news/veranstaltungskalender)

- Joker Ausgabezahlen auf die Bewertungen von Gerichten matchen und versuchen einen Nachfragetrend daraus abzuleiten

# Feature Engineering Ideen
