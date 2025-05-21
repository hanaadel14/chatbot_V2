import zipfile
from pathlib import Path

# now assume data.zip is in the same folder as this script
ROOT = Path(__file__).parent
ZIP  = ROOT / "data.zip"
OUT  = ROOT.parent / "data_raw"      # project_root/data_raw

OUT.mkdir(exist_ok=True)

with zipfile.ZipFile(ZIP, "r") as zf:
    zf.extractall(OUT)

print(f"Extracted {len(zf.namelist())} files to {OUT}")
