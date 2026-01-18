"""
Lightweight history store with CSV persistence (Excel-friendly).
"""
import csv
import datetime

from soil_segment.config import HISTORY_FILE
HISTORY_FIELDS = [
    "id",
    "name",
    "lot_number",
    "formula",
    "threshold",
    "total_images",
    "passed_images",
    "date",
    "n",
    "p",
    "k",
    "status",
]


def _parse_float(value, default=None):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _parse_int(value, default=None):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def load_history(limit=25):
    """
    Load history from CSV into a list for the API.
    Returns (items, max_id).
    """
    if not HISTORY_FILE.exists():
        return [], 0

    rows = []
    max_id = 0

    try:
        with HISTORY_FILE.open("r", newline="") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                item_id = _parse_int(row.get("id")) or len(rows) + 1
                max_id = max(max_id, item_id)
                rows.append({
                    "id": item_id,
                    "name": row.get("name") or "upload",
                    "lot_number": row.get("lot_number") or "N/A",
                    "formula": row.get("formula") or "N/A",
                    "threshold": _parse_float(row.get("threshold")),
                    "total_images": _parse_int(row.get("total_images"), 0) or 0,
                    "passed_images": _parse_int(row.get("passed_images"), 0) or 0,
                    "date": row.get("date") or datetime.datetime.now().strftime("%Y-%m-%d"),
                    "n": _parse_float(row.get("n"), 0.0),
                    "p": _parse_float(row.get("p"), 0.0),
                    "k": _parse_float(row.get("k"), 0.0),
                    "status": row.get("status") or "ok",
                })
    except Exception as exc:
        print(f"Warning: failed to load history CSV: {exc}")
        return [], 0

    return rows[-limit:], max_id


def append_history(record):
    """Append a history record to CSV for persistence."""
    try:
        HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
        write_header = not HISTORY_FILE.exists()
        with HISTORY_FILE.open("a", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=HISTORY_FIELDS)
            if write_header:
                writer.writeheader()
            writer.writerow({
                "id": record.get("id", ""),
                "name": record.get("name", ""),
                "lot_number": record.get("lot_number", ""),
                "formula": record.get("formula", ""),
                "threshold": record.get("threshold", ""),
                "total_images": record.get("total_images", ""),
                "passed_images": record.get("passed_images", ""),
                "date": record.get("date", ""),
                "n": record.get("n", ""),
                "p": record.get("p", ""),
                "k": record.get("k", ""),
                "status": record.get("status", ""),
            })
    except Exception as exc:
        print(f"Warning: failed to write history CSV: {exc}")
