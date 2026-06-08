from __future__ import annotations

import re
import zipfile
from datetime import date
from io import BytesIO
from pathlib import Path
from typing import Iterable
from xml.etree import ElementTree as ET

import matplotlib.pyplot as plt
import pandas as pd
import requests


START_DATE = date(2025, 1, 1)
END_DATE = date.today()

# DWD CAP warning product (district status, German language).
BASE_CAP_URL = "https://opendata.dwd.de/weather/alerts/cap/DISTRICT_DWD_STAT/"
CAP_FILE_RE = re.compile(
    r'href="(Z_CAP_C_EDZW_(\d{14})_PVW_STATUS_PREMIUMDWD_DISTRICT_DE\.zip)"'
)

TARGET_WARNCELL_ID = "914524002"


def list_cap_files_since(start_date: date, end_date: date) -> list[str]:
    response = requests.get(BASE_CAP_URL, timeout=60)
    response.raise_for_status()

    files: list[str] = []
    for match in CAP_FILE_RE.finditer(response.text):
        filename = match.group(1)
        timestamp = pd.to_datetime(match.group(2), format="%Y%m%d%H%M%S", utc=True)
        if start_date <= timestamp.date() <= end_date:
            files.append(filename)

    files.sort()
    return files


def _text(node: ET.Element | None, tag: str) -> str | None:
    if node is None:
        return None
    found = node.find(f"{{*}}{tag}")
    if found is None or found.text is None:
        return None
    return found.text.strip() or None


def _iter_info_nodes(root: ET.Element) -> Iterable[ET.Element]:
    return root.findall(".//{*}info")


def _normalize_code(code: str) -> str:
    return "".join(ch for ch in code if ch.isdigit())


def _area_has_target_code(area: ET.Element, target_codes: set[str]) -> tuple[bool, list[str]]:
    codes = [
        _normalize_code(value.text.strip())
        for value in area.findall(".//{*}geocode/{*}value")
        if value.text and _normalize_code(value.text.strip())
    ]
    return any(code in target_codes for code in codes), codes


def warning_records_from_zip(file_url: str, target_codes: set[str]) -> list[dict[str, str | None]]:
    response = requests.get(file_url, timeout=60)
    response.raise_for_status()

    records: list[dict[str, str | None]] = []
    with zipfile.ZipFile(BytesIO(response.content)) as zf:
        for member_name in zf.namelist():
            if not member_name.lower().endswith(".xml"):
                continue
            xml_bytes = zf.read(member_name)
            root = ET.fromstring(xml_bytes)

            identifier = _text(root, "identifier")
            sent = _text(root, "sent")
            status = _text(root, "status")
            msg_type = _text(root, "msgType")

            for info in _iter_info_nodes(root):
                language = _text(info, "language")
                if language and not language.lower().startswith("de"):
                    continue

                matched_area_desc: list[str] = []
                matched_codes: list[str] = []
                for area in info.findall(".//{*}area"):
                    is_match, area_codes = _area_has_target_code(area, target_codes)
                    if not is_match:
                        continue
                    area_desc = _text(area, "areaDesc")
                    if area_desc:
                        matched_area_desc.append(area_desc)
                    matched_codes.extend(area_codes)

                if not matched_area_desc and not matched_codes:
                    continue

                records.append(
                    {
                        "identifier": identifier,
                        "sent": sent,
                        "status": status,
                        "msg_type": msg_type,
                        "event": _text(info, "event"),
                        "severity": _text(info, "severity"),
                        "urgency": _text(info, "urgency"),
                        "certainty": _text(info, "certainty"),
                        "onset": _text(info, "onset"),
                        "expires": _text(info, "expires"),
                        "headline": _text(info, "headline"),
                        "area_desc": " | ".join(sorted(set(matched_area_desc))),
                        "warncell_ids": " | ".join(sorted(set(matched_codes))),
                    }
                )

    return records


def build_warning_frame(files: list[str], target_warncell_id: str) -> pd.DataFrame:
    target_codes = {_normalize_code(target_warncell_id)}
    all_records: list[dict[str, str | None]] = []
    for idx, filename in enumerate(files, start=1):
        if idx % 50 == 0 or idx == 1 or idx == len(files):
            print(f"Processing CAP file {idx}/{len(files)}: {filename}")
        all_records.extend(
            warning_records_from_zip(
                file_url=BASE_CAP_URL + filename,
                target_codes=target_codes,
            )
        )

    if not all_records:
        return pd.DataFrame(
            columns=[
                "identifier",
                "sent",
                "status",
                "msg_type",
                "event",
                "severity",
                "urgency",
                "certainty",
                "onset",
                "expires",
                "headline",
                "area_desc",
                "warncell_ids",
            ]
        )

    df = pd.DataFrame(all_records).drop_duplicates()
    for dt_col in ["sent", "onset", "expires"]:
        df[dt_col] = pd.to_datetime(df[dt_col], utc=True, errors="coerce")

    df = df[df["sent"].dt.date >= START_DATE].copy()
    df.sort_values(["sent", "identifier"], inplace=True)
    return df


def plot_warnings(df: pd.DataFrame, output_path: Path, location_label: str) -> None:
    active = df[df["msg_type"].fillna("").str.lower() != "cancel"].copy()

    monthly_counts = (
        active.assign(month=active["sent"].dt.to_period("M").dt.to_timestamp())
        .groupby("month")
        .size()
        .rename("warnings")
        .reset_index()
    )
    event_counts = active["event"].fillna("Unknown").value_counts().head(12).sort_values(ascending=True)

    fig, axes = plt.subplots(2, 1, figsize=(13, 9))

    axes[0].bar(monthly_counts["month"], monthly_counts["warnings"], color="#4c78a8")
    axes[0].set_title(f"DWD Warnings per Month since 2025 ({location_label})")
    axes[0].set_ylabel("Count")
    axes[0].set_xlabel("")

    axes[1].barh(event_counts.index, event_counts.values, color="#f58518")
    axes[1].set_title(f"Most Frequent Warning Types ({location_label})")
    axes[1].set_xlabel("Count")
    axes[1].set_ylabel("")

    fig.autofmt_xdate()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def main() -> None:
    target_warncell_id = TARGET_WARNCELL_ID
    location_label = f"WARNCELLID {target_warncell_id}"

    cap_files = list_cap_files_since(START_DATE, END_DATE)
    print(f"CAP files found in directory since {START_DATE.isoformat()}: {len(cap_files)}")

    warnings_df = build_warning_frame(cap_files, target_warncell_id=target_warncell_id)
    print(f"Warnings found for {location_label}: {len(warnings_df)}")
    print(f"CAP geocode filter: {target_warncell_id}")

    output_root = Path(__file__).resolve().parents[1]
    csv_output = output_root / "data" / "dwd_warnings_trier_since_2025.csv"
    plot_output = output_root / "plots" / "dwd_warnings_trier_since_2025.png"

    csv_output.parent.mkdir(parents=True, exist_ok=True)
    warnings_df.to_csv(csv_output, index=False)
    if not warnings_df.empty:
        plot_warnings(warnings_df, plot_output, location_label=location_label)
        print(f"Saved warning visualization to: {plot_output}")
    else:
        print("No warnings found for the selected location and period.")

    print(f"Saved warning list to: {csv_output}")
    print("\nPreview:")
    print(warnings_df.head(20))


if __name__ == "__main__":
    main()
