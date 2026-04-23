# prepare_data.py
import pandas as pd
import re
from pathlib import Path

DATA_DIR = Path("data")
OUT_FILE = DATA_DIR / "processed_training.csv"

def clean_text(s):
    if pd.isna(s): return ""
    s = str(s).lower()
    s = re.sub(r'[^a-z0-9,; ]+', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def combine_symptom_cols(df):
    # Auto-detect columns starting with 'Symptom' or 'symptom' or a 'Symptoms' column
    symptom_cols = [c for c in df.columns if c.lower().startswith("symptom")]
    if "Symptoms" in df.columns:
        symptom_cols.append("Symptoms")
    symptom_cols = [c for c in symptom_cols if c in df.columns]
    if symptom_cols:
        df['symptoms'] = df[symptom_cols].astype(str).apply(
            lambda row: ", ".join([x for x in row if x and str(x).lower() != 'nan']),
            axis=1
        )
    return df

def main():
    df_list = []
    # load DiseaseAndSymptoms.csv
    p1 = DATA_DIR / "DiseaseAndSymptoms.csv"
    p2 = DATA_DIR / "Diseases_Symptoms.csv"
    p3 = DATA_DIR / "Disease precaution.csv"
    for p in [p1, p2]:
        if p.exists():
            df = pd.read_csv(p, encoding='utf-8', low_memory=False)
            # find disease column
            disease_col = None
            for col in df.columns:
                if col.lower() in ("disease", "name", "diseases"):
                    disease_col = col; break
            if disease_col is None:
                # fallback to first string column
                for col in df.columns:
                    if df[col].dtype == object:
                        disease_col = col; break
            df = combine_symptom_cols(df)
            if 'symptoms' not in df.columns:
                # Try 'Symptom_1' etc detection
                symptom_cols = [c for c in df.columns if 'symptom' in c.lower()]
                if symptom_cols:
                    df['symptoms'] = df[symptom_cols].astype(str).apply(
                        lambda row: ", ".join([x for x in row if x and str(x).lower()!='nan']), axis=1)
            df = df[[disease_col, 'symptoms']].dropna(subset=['symptoms'])
            df.columns = ['disease', 'symptoms']
            df['disease'] = df['disease'].astype(str).str.strip()
            df['symptoms'] = df['symptoms'].astype(str).apply(clean_text)
            df_list.append(df)
    if not df_list:
        raise FileNotFoundError("No symptom files found in data/ (DiseaseAndSymptoms.csv or Diseases_Symptoms.csv)")
    combined = pd.concat(df_list, ignore_index=True).drop_duplicates().reset_index(drop=True)

    # Optionally include Disease precaution if it has disease + precaution text
    if p3.exists():
        df3 = pd.read_csv(p3, encoding='utf-8', low_memory=False)
        if df3.shape[1] >= 1:
            # try first column as disease, combine precaution columns
            cols = df3.columns.tolist()
            disease_col = cols[0]
            other_cols = cols[1:]
            if other_cols:
                df3['symptoms'] = df3[other_cols].astype(str).apply(
                    lambda row: ", ".join([x for x in row if x and str(x).lower()!='nan']), axis=1)
                df3 = df3[[disease_col, 'symptoms']].dropna()
                df3.columns = ['disease', 'symptoms']
                df3['symptoms'] = df3['symptoms'].astype(str).apply(clean_text)
                combined = pd.concat([combined, df3], ignore_index=True).drop_duplicates().reset_index(drop=True)

    # write processed file
    combined.to_csv(OUT_FILE, index=False)
    print(f"Wrote processed training file: {OUT_FILE}")
    print("Rows:", combined.shape[0])

if __name__ == "__main__":
    main()
