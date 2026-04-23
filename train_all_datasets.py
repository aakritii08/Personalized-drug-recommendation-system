
import pandas as pd
import numpy as np
import os, re, pickle, warnings
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from rapidfuzz import process, fuzz  # 🔍 for fuzzy name matching

# === Suppress warnings ===
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

print("🚀 Running FINAL ENHANCED VERSION of train_all_datasets.py")

# === Paths ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# === Utility ===
def clean_text(s):
    if pd.isna(s):
        return ""
    s = str(s).lower()
    s = re.sub(r'[^a-z0-9,; ]+', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

# === Load datasets ===
files = {
    "sym1": "DiseaseAndSymptoms.csv",
    "sym2": "Diseases_Symptoms.csv",
    "precaution": "Disease precaution.csv",
    "medicine": "Medicine_Details.csv"
}

dfs = {}
for key, filename in files.items():
    path = os.path.join(DATA_DIR, filename)
    if os.path.exists(path):
        try:
            dfs[key] = pd.read_csv(path, encoding="utf-8", low_memory=False)
            print(f"✅ Loaded: {filename} ({dfs[key].shape[0]} rows)")
        except Exception as e:
            print(f"⚠️ Error reading {filename}: {e}")
    else:
        print(f"❌ Missing file: {filename}")

# === Combine Disease-Symptom Data ===
train_rows = []

if "sym1" in dfs:
    df = dfs["sym1"].copy()
    disease_col = [c for c in df.columns if "disease" in c.lower()][0]
    symptom_cols = [c for c in df.columns if "symptom" in c.lower()]
    df["symptoms"] = df[symptom_cols].astype(str).apply(
        lambda row: ", ".join([x for x in row if x and str(x).lower() != "nan"]), axis=1)
    df = df[[disease_col, "symptoms"]].dropna()
    df.columns = ["disease", "symptoms"]
    train_rows.append(df)

if "sym2" in dfs:
    df2 = dfs["sym2"].copy()
    possible_disease_col = [c for c in df2.columns if "name" in c.lower() or "disease" in c.lower()]
    possible_symptom_col = [c for c in df2.columns if "symptom" in c.lower()]
    if possible_disease_col and possible_symptom_col:
        df2 = df2[[possible_disease_col[0], possible_symptom_col[0]]]
        df2.columns = ["disease", "symptoms"]
        train_rows.append(df2)

if "precaution" in dfs:
    df3 = dfs["precaution"].copy()
    cols = df3.columns.tolist()
    if len(cols) > 1:
        disease_col = cols[0]
        df3["symptoms"] = df3[cols[1:]].astype(str).apply(
            lambda row: ", ".join([x for x in row if x and str(x).lower() != "nan"]), axis=1)
        df3 = df3[[disease_col, "symptoms"]]
        df3.columns = ["disease", "symptoms"]
        train_rows.append(df3)

train_df = pd.concat(train_rows, ignore_index=True)
train_df.dropna(inplace=True)
train_df["disease"] = train_df["disease"].astype(str).str.strip()
train_df["symptoms"] = train_df["symptoms"].astype(str).apply(clean_text)
train_df.drop_duplicates(inplace=True)

print(f"🧾 Combined training dataset: {len(train_df)} rows, {train_df['disease'].nunique()} unique diseases")

# === Encode & Train ===
X = train_df["symptoms"]
y = train_df["disease"]

vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)
X_vec = vectorizer.fit_transform(X)

le = LabelEncoder()
y_enc = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X_vec, y_enc, test_size=0.2, random_state=42)

models = {
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
    "Logistic Regression": LogisticRegression(max_iter=500),
    "Naive Bayes": MultinomialNB()
}

print("\n🔍 Model Performance Comparison:\n")
model_scores = {}
for name, clf in models.items():
    clf.fit(X_train, y_train)
    acc = clf.score(X_test, y_test)
    model_scores[name] = acc
    print(f"{name:20s} → Accuracy: {acc:.4f}")

best_model_name = max(model_scores, key=model_scores.get)
model = models[best_model_name]
print(f"\n🏆 Best model selected: {best_model_name} (Accuracy: {model_scores[best_model_name]:.4f})")

y_pred = model.predict(X_test)
report = classification_report(y_test, y_pred, zero_division=0)
acc = accuracy_score(y_test, y_pred)
print(f"✅ Final Model Accuracy: {acc:.4f}")

with open(os.path.join(MODEL_DIR, "model_report.txt"), "w", encoding="utf-8") as f:
    f.write(report)
print("📄 Classification report saved as model_report.txt")

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, cmap="Blues", xticklabels=False, yticklabels=False)
plt.title(f"Confusion Matrix — {best_model_name}")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, "confusion_matrix.png"))
plt.close()
print("📊 Confusion matrix saved as confusion_matrix.png")

pickle.dump(model, open(os.path.join(MODEL_DIR, "model.pkl"), "wb"))
pickle.dump(vectorizer, open(os.path.join(MODEL_DIR, "vectorizer.pkl"), "wb"))
pickle.dump(le, open(os.path.join(MODEL_DIR, "label_encoder.pkl"), "wb"))
train_df.to_csv(os.path.join(MODEL_DIR, "training_samples_full.csv"), index=False)
print("📦 Model & training data saved successfully!")

# === Combine Medicine + Precaution Info ===
medicine_df = dfs.get("medicine", pd.DataFrame()).copy()
precaution_df = dfs.get("precaution", pd.DataFrame()).copy()

if not medicine_df.empty:
    medicine_df.columns = [c.strip().capitalize() for c in medicine_df.columns]
if not precaution_df.empty:
    precaution_df.columns = [c.strip().capitalize() for c in precaution_df.columns]

combined_info = {}
for i, disease in enumerate(train_df["disease"].unique(), 1):
    print(f"Processing {i}/{len(train_df['disease'].unique())}: {disease}")
    info = {}
    meds = []
    if not medicine_df.empty:
        mask = medicine_df.apply(lambda row: row.astype(str).str.contains(disease, case=False, regex=False).any(), axis=1)
        matched = medicine_df[mask]
        for _, row in matched.iterrows():
            meds.append({
                "Medicine": row.get("Medicine name", ""),
                "Uses": row.get("Uses", ""),
                "Side_effects": row.get("Side_effects", ""),
                "Composition": row.get("Composition", "")
            })
    info["medicines"] = meds

    if not precaution_df.empty and "Disease" in precaution_df.columns:
        precs = precaution_df[precaution_df["Disease"].str.lower() == disease.lower()]
        if not precs.empty:
            prec_list = [val for val in precs.iloc[0, 1:].tolist() if pd.notna(val)]
            info["precautions"] = prec_list
    combined_info[disease] = info

with open(os.path.join(MODEL_DIR, "disease_info.pkl"), "wb") as f:
    pickle.dump(combined_info, f)
print("\n🩺 Combined medicine + precaution info saved (disease_info.pkl)")

# === Sentiment Integration (with fuzzy matching) ===
sentiment_path = os.path.join(DATA_DIR, "Drug_Reviews_Sentiment.csv")
if os.path.exists(sentiment_path):
    sentiment_df = pd.read_csv(sentiment_path, encoding="utf-8", low_memory=False)
    print(f"✅ Loaded: Drug_Reviews_Sentiment.csv ({sentiment_df.shape[0]} rows)")

    sentiment_df.columns = [c.strip().lower().replace(" ", "_") for c in sentiment_df.columns]
    if all(col in sentiment_df.columns for col in ["drugname", "rating", "condition"]):
        sentiment_df["rating"] = pd.to_numeric(sentiment_df["rating"], errors="coerce").fillna(0)
        sentiment_df["sentiment_score"] = sentiment_df["rating"] / 10

        def safe_top_condition(x):
            vals = x.value_counts()
            return vals.index[0] if len(vals) > 0 else "Unknown"

        sentiment_summary = (
            sentiment_df.groupby("drugname")
            .agg(
                avg_rating=("rating", "mean"),
                avg_sentiment=("sentiment_score", "mean"),
                total_reviews=("review", "count"),
                avg_useful_count=("usefulcount", "mean"),
                top_condition=("condition", safe_top_condition)
            )
            .reset_index()
        )

        sentiment_summary["avg_rating"] = sentiment_summary["avg_rating"].round(2)
        sentiment_summary["avg_sentiment"] = (sentiment_summary["avg_sentiment"] * 100).round(1)
        sentiment_summary.to_csv(os.path.join(MODEL_DIR, "medicine_sentiment_summary.csv"), index=False)
        print(f"✅ Sentiment summary created for {len(sentiment_summary)} medicines.")

        if not medicine_df.empty:
            med_copy = medicine_df.copy()
            med_copy.columns = [c.strip().lower().replace(" ", "_") for c in med_copy.columns]
            merge_key = "medicine_name" if "medicine_name" in med_copy.columns else "medicine"

            # === Prepare clean names ===
            med_copy["clean_med_name"] = med_copy[merge_key].str.replace(r"[^a-zA-Z0-9 ]", "", regex=True).str.lower().str.strip()
            sentiment_summary["clean_drugname"] = sentiment_summary["drugname"].str.replace(r"[^a-zA-Z0-9 ]", "", regex=True).str.lower().str.strip()

            print("🔍 Performing fuzzy name matching between medicine and sentiment datasets...")
            sentiment_map = {}
            all_drug_names = sentiment_summary["clean_drugname"].tolist()

            for med in med_copy["clean_med_name"].unique():
                best_match = process.extractOne(med, all_drug_names, scorer=fuzz.token_sort_ratio)
                if best_match and best_match[1] >= 80:
                    sentiment_map[med] = best_match[0]

            med_copy["matched_drugname"] = med_copy["clean_med_name"].map(sentiment_map)
            merged_df = pd.merge(
                med_copy,
                sentiment_summary,
                left_on="matched_drugname",
                right_on="clean_drugname",
                how="left"
            )

            merged_df.to_csv(os.path.join(MODEL_DIR, "medicine_with_sentiment.csv"), index=False)
            print("📊 Fuzzy merged sentiment data saved (medicine_with_sentiment.csv)")
    else:
        print("⚠️ Missing expected columns (drugName, rating, condition). Skipping sentiment merge.")
else:
    print("❌ Sentiment dataset not found. Skipping sentiment integration.")

print("\n✅ Training & Evaluation complete!")
