from pathlib import Path
import pyreadr

ADNI_RDA_DIR = Path("/Users/sanghati/Downloads/ADNIMERGE2/data")

FILES = {
    "dxsum": ADNI_RDA_DIR / "DXSUM.rda",
    "visits": ADNI_RDA_DIR / "VISITS.rda",
    "ptdemog": ADNI_RDA_DIR / "PTDEMOG.rda",
    "mmse": ADNI_RDA_DIR / "MMSE.rda",
    "adas": ADNI_RDA_DIR / "ADAS.rda",
    "cdr": ADNI_RDA_DIR / "CDR.rda",
}

def read_rda(path):
    result = pyreadr.read_r(str(path))
    name = list(result.keys())[0]
    return result[name]

def main():
    print("=== BUILD ADNI CORE TABLE ===")

    for name, path in FILES.items():
        if not path.exists():
            print(f"Missing: {path}")
            return

        df = read_rda(path)
        print(f"{name}: {df.shape[0]} rows, {df.shape[1]} columns")
        print(df.columns[:10].tolist())
        print("-" * 50)

if __name__ == "__main__":
    main()