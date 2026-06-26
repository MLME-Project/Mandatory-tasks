import sys
from pathlib import Path
from typing import List

try:
    import pandas as pd
except ImportError:
    print("Pandas required: pip install pandas openpyxl")
    sys.exit(1)

from task01_visualize import visualize_task01_results


def _maybe_float(val):
    try:
        if pd.isna(val):
            raise ValueError
        return float(val)
    except Exception:
        raise


def normalize_filename(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch.isalnum())


def find_existing_file(requested_path: str, base_dir: Path) -> Path | None:
    requested = Path(requested_path)
    if requested.is_absolute() and requested.exists():
        return requested

    possible = base_dir / requested
    if possible.exists():
        return possible.resolve()

    normalized_requested = normalize_filename(requested.name)
    for child in base_dir.iterdir():
        if child.is_file() and normalize_filename(child.name) == normalized_requested:
            return child.resolve()
    return None


def read_summary_excel(path: str) -> List[dict]:
    xls = pd.ExcelFile(path)
    rows = []
    for sheet in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name=sheet, header=0)
        for idx in range(len(df)):
            try:
                scale = str(df.iloc[idx, 2]).strip()
                t = _maybe_float(df.iloc[idx, 3])
                ph = _maybe_float(df.iloc[idx, 4])
                f1 = _maybe_float(df.iloc[idx, 5])
                f2 = _maybe_float(df.iloc[idx, 6])
                f3 = _maybe_float(df.iloc[idx, 7])
                y_val = None
                if 8 in df.columns or len(df.columns) > 8:
                    try:
                        y_val = _maybe_float(df.iloc[idx, 8])
                    except Exception:
                        y_val = None
                if y_val is None:
                    for candidate in ("Y", "y", "yield", "quality", "score"):
                        if candidate in df.columns:
                            try:
                                y_val = _maybe_float(df.at[idx, candidate])
                                break
                            except Exception:
                                y_val = None

                if y_val is None:
                    continue

                rows.append({
                    "scale": scale,
                    "T": t,
                    "pH": ph,
                    "F1": f1,
                    "F2": f2,
                    "F3": f3,
                    "quality": y_val,
                    "status": "ok",
                    "result": {"Y": y_val},
                })
            except Exception:
                continue
    return rows


if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent
    path = "Summery_Versuche.xlsx"
    if len(sys.argv) > 1:
        path = " ".join(sys.argv[1:]).strip('"')
    file_path = find_existing_file(path, script_dir)
    if file_path is None:
        print(f"Excel file nicht gefunden: {path}")
        print(f"Aktuelles Skriptverzeichnis: {script_dir}")
        print("Suche nach ähnlichen Dateien im selben Ordner...")
        candidates = [child.name for child in script_dir.iterdir() if child.is_file()]
        for candidate in candidates:
            print(f"  - {candidate}")
        sys.exit(1)

    print(f"Verwende Excel-Datei: {file_path}")
    data = read_summary_excel(str(file_path))
    print(f"Parsed {len(data)} valid rows from {file_path}")
    if not data:
        print("No data to visualize.")
        sys.exit(0)
    visualize_task01_results(data)
