from flask import Flask, render_template, request, flash
import pandas as pd
import re
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from difflib import get_close_matches

app = Flask(__name__)
app.secret_key = "replace-with-a-secure-key"

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent / "data"

# --- Utility ---
def safe_read_csv(path):
    try:
        df = pd.read_csv(path)
        print(f"✅ Loaded {path.name} with {len(df)} rows")
        return df
    except Exception as e:
        print(f"⚠️ Error reading {path.name}: {e}")
        return pd.DataFrame()

def clean_text(s):
    s = str(s).lower()
    s = re.sub(r'[^a-z0-9 ,;]+', ' ', s)
    return s.strip()

# --- Load datasets ---
symptom_df = safe_read_csv(DATA_DIR / "Diseases_Symptoms.csv")
precaution_df = safe_read_csv(DATA_DIR / "Disease precaution.csv")
medicine_df = safe_read_csv(DATA_DIR / "Medicine_Details.csv")
sentiment_df = safe_read_csv(DATA_DIR / "Drug_Reviews_Sentiment.csv")  # ✅ new dataset

# --- Preprocess symptoms ---
if not symptom_df.empty:
    disease_col = next((c for c in symptom_df.columns if "disease" in c.lower() or "name" in c.lower()), None)
    symptom_cols = [c for c in symptom_df.columns if "symptom" in c.lower()]
    if disease_col and symptom_cols:
        symptom_df["Symptoms"] = symptom_df[symptom_cols].astype(str).apply(
            lambda r: ", ".join([x for x in r if x.lower() != "nan" and x.strip() != ""]),
            axis=1
        )
        symptom_df = symptom_df[[disease_col, "Symptoms"]]
        symptom_df.columns = ["Disease", "Symptoms"]
    else:
        symptom_df = symptom_df.iloc[:, :2]
        symptom_df.columns = ["Disease", "Symptoms"]
    symptom_df["Symptoms"] = symptom_df["Symptoms"].apply(clean_text)
else:
    symptom_df = pd.DataFrame(columns=["Disease", "Symptoms"])

# --- TF-IDF setup ---
if not symptom_df.empty:
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(symptom_df["Symptoms"])
else:
    vectorizer = None
    tfidf_matrix = None

# --- Clean data ---
medicine_df.fillna("", inplace=True)
precaution_df.columns = [c.strip().capitalize() for c in precaution_df.columns]

# --- Clean sentiment data ---
if not sentiment_df.empty:
    sentiment_df.columns = [c.strip().capitalize() for c in sentiment_df.columns]
    if "Sentiment" in sentiment_df.columns:
        sentiment_df["Sentiment"] = sentiment_df["Sentiment"].astype(str).str.lower().map({
            "positive": 1.0, "neutral": 0.5, "negative": 0.0
        }).fillna(0.5)
else:
    sentiment_df = pd.DataFrame()

# --- Recommendation Function ---
def recommend_drugs(symptom_input):
    if vectorizer is None or tfidf_matrix is None or symptom_df.empty:
        return {"diseases": [], "medicines": [], "precautions": {}}

    query = clean_text(symptom_input)
    query_vec = vectorizer.transform([query])
    sims = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_idx = sims.argsort()[::-1][:5]
    top_diseases = [(symptom_df.iloc[i]["Disease"], round(float(sims[i]), 3)) for i in top_idx if sims[i] > 0]

    results = {"diseases": top_diseases, "medicines": [], "precautions": {}}
    precaution_names = [str(x).lower() for x in precaution_df["Disease"]] if "Disease" in precaution_df.columns else []

    for disease, score in top_diseases:
        disease_lower = disease.lower()

        # --- MEDICINES ---
        meds = medicine_df[
            medicine_df.apply(lambda row: any(disease_lower in str(v).lower() for v in row.values), axis=1)
        ].copy()
        meds["match_score"] = score

        # --- Add sentiment ---
        if not sentiment_df.empty and "Medicine name" in sentiment_df.columns:
            for idx, med_row in meds.iterrows():
                med_name = med_row.get("Medicine Name", "")
                if med_name:
                    matched_sent = sentiment_df[sentiment_df["Medicine name"].str.lower() == med_name.lower()]
                    if not matched_sent.empty:
                        avg_sent = matched_sent["Sentiment"].mean()
                        meds.at[idx, "Avg_Sentiment"] = round(avg_sent, 2)
                    else:
                        meds.at[idx, "Avg_Sentiment"] = None
                else:
                    meds.at[idx, "Avg_Sentiment"] = None
        else:
            meds["Avg_Sentiment"] = None

        results["medicines"].extend(meds.to_dict(orient="records"))

        # --- PRECAUTIONS (fuzzy + substring) ---
        if "Disease" in precaution_df.columns:
            best_match = get_close_matches(disease_lower, precaution_names, n=1, cutoff=0.5)
            if best_match:
                match_name = best_match[0]
                prec_row = precaution_df[precaution_df["Disease"].str.lower() == match_name]
            else:
                prec_row = precaution_df[precaution_df["Disease"].str.lower().str.contains(disease_lower[:5], na=False)]
            if not prec_row.empty:
                precs = [v for v in prec_row.iloc[0, 1:].tolist() if str(v).strip() not in ("", "nan")]
                results["precautions"][disease] = precs if precs else ["No specific precautions available."]
            else:
                results["precautions"][disease] = ["No specific precautions available."]

    # --- Sort & Limit ---
    if "Average Review %" in medicine_df.columns:
        for m in results["medicines"]:
            try:
                m["Average Review %"] = float(m.get("Average Review %", 0))
            except:
                m["Average Review %"] = 0.0
        results["medicines"] = sorted(
            results["medicines"],
            key=lambda x: (x["match_score"], x.get("Avg_Sentiment", 0), x["Average Review %"]),
            reverse=True
        )

    results["medicines"] = results["medicines"][:10]
    return results

# --- Flask Routes ---
@app.route("/", methods=["GET", "POST"])
def index():
    results = None
    symptoms = ""
    if request.method == "POST":
        symptoms = request.form.get("symptoms", "")
        if not symptoms.strip():
            flash("Please enter some symptoms.")
        else:
            results = recommend_drugs(symptoms)
            if not results["medicines"]:
                flash("No medicines found for the entered symptoms.")
    return render_template("index.html", results=results, symptoms=symptoms)

@app.route("/about")
def about():
    return render_template("about.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8501, debug=True)
