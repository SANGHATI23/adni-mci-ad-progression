from pathlib import Path
import re
import pyreadr
import pandas as pd

ADNI_RDA_DIR = Path("/Users/sanghati/Downloads/ADNIMERGE2/data")
DXSUM_FILE = ADNI_RDA_DIR / "DXSUM.rda"

OUTPUT_DIR = Path("data/processed")
OUTPUT_FILE = OUTPUT_DIR / "mci_conversion_labels.csv"


def read_rda(path):
    result = pyreadr.read_r(str(path))
    name = list(result.keys())[0]
    return result[name]


def visit_month(viscode):
    if pd.isna(viscode):
        return None
    viscode = str(viscode).lower()

    if viscode in ["bl", "sc", "scmri"]:
        return 0

    match = re.search(r"m(\d+)", viscode)
    if match:
        return int(match.group(1))

    return None


def main():
    print("=== MCI CONVERSION LABELING ===")

    df = read_rda(DXSUM_FILE)

    df = df[["RID", "VISCODE", "EXAMDATE", "DIAGNOSIS"]].copy()
    df = df.dropna(subset=["RID", "VISCODE", "DIAGNOSIS"])

    df["visit_month"] = df["VISCODE"].apply(visit_month)
    df = df.dropna(subset=["visit_month"])

    df["EXAMDATE"] = pd.to_datetime(df["EXAMDATE"], errors="coerce")
    df = df.sort_values(["RID", "visit_month", "EXAMDATE"])

    labels = []

    for rid, patient in df.groupby("RID"):
        patient = patient.sort_values("visit_month")

        baseline = patient[patient["visit_month"] == 0]
        if baseline.empty:
            continue

        baseline_dx = baseline.iloc[0]["DIAGNOSIS"]

        if baseline_dx != "MCI":
            continue

        followup = patient[patient["visit_month"] > 0]

        converted = (followup["DIAGNOSIS"] == "Dementia").any()

        conversion_month = None
        if converted:
            conversion_month = followup.loc[
                followup["DIAGNOSIS"] == "Dementia", "visit_month"
            ].min()

        max_followup_month = followup["visit_month"].max() if not followup.empty else 0

        # STRICT 36-MONTH COHORT

        if converted and conversion_month <= 36:
           label = "pMCI_36"

        elif (not converted) and max_followup_month >= 36:
           label = "sMCI_36"

        else:
            continue  # discard weak cases

        labels.append({
            "RID": rid,
            "baseline_diagnosis": baseline_dx,
            "label": label,
            "converted_to_dementia": converted,
            "conversion_month": conversion_month,
            "max_followup_month": max_followup_month,
            "num_visits": len(patient)
        })

    out = pd.DataFrame(labels)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_FILE, index=False)

    print(f"Saved: {OUTPUT_FILE}")
    print(f"Rows: {out.shape[0]}")
    print("\nLabel counts:")
    print(out["label"].value_counts(dropna=False))

    print("\nFirst 10 rows:")
    print(out.head(10))


if __name__ == "__main__":
    main()