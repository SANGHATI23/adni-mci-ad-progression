from pathlib import Path
import pandas as pd

RAW_DIR = Path("data/raw/adni")
ADNIMERGE_FILE = RAW_DIR / "ADNIMERGE.csv"

def main():
    print("=== ADNI REAL DATA LOADER ===")

    if not ADNIMERGE_FILE.exists():
        print(f"Missing file: {ADNIMERGE_FILE}")
        print("Download ADNIMERGE.csv from ADNI/LONI and place it here:")
        print(RAW_DIR)
        return

    df = pd.read_csv(ADNIMERGE_FILE, low_memory=False)

    print(f"Loaded: {ADNIMERGE_FILE}")
    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")

    print("\nFirst 10 columns:")
    print(df.columns[:10].tolist())

    print("\nFirst 5 rows:")
    print(df.head())

if __name__ == "__main__":
    main()
