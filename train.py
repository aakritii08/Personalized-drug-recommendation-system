# train_model_full.py
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle
import os

# ============================
# 1️⃣ Load all datasets
# ============================

# Adjust paths if needed
base_dir = os.path.dirname(os.path.abspath(__file__))
data_files = [
    "data/DiseaseAndSymptoms.csv",
    "data/Diseases_Symptoms.csv",
    "data/Disease precaution.csv"
]

dfs = []
for file in data_files:
    path = os.path.join(base_dir, file)
    if os.path.exists(path):
        print(f"✅ Loaded {file}")
        dfs.append(pd.read_csv(path))
    else:
        print(f"⚠️ Missing file: {file}")

# Merge dataframes that have symptom/disease columns
df = pd.concat(dfs, ignore_index=True)
df = df.rename(columns={col: col.strip().capitalize() for col in df.columns})

# Try to find disease and symptom columns automatically
possible_disease_cols = [c for c in df.columns if 'disease' in c.lower()]
possible_symptom_cols = [c for c in df.columns if 'symptom' in c.lower()]

if not possible_disease_cols or not possible_symptom_cols:
    raise ValueError("Could not find disease or symptom columns in dataset.")

disease_col = possible_disease_cols[0]
symptom_col = possible_symptom_cols[0]

print(f"✅ Using columns: Disease = '{disease_col}', Symptoms = '{symptom_col}'")

# Clean data
df = df.dropna(subset=[disease_col, symptom_col])
df = df.drop_duplicates(subset=[symptom_col])

print(f"Total records used for training: {len(df)}")

# ============================
# 2️⃣ Encode data
# ============================

X = df[symptom_col].astype(str)
y = df[disease_col].astype(str)

vectorizer = TfidfVectorizer(max_features=3000, stop_words='english')
X_vec = vectorizer.fit_transform(X)

label_encoder = LabelEncoder()
y_enc = label_encoder.fit_transform(y)

# ============================
# 3️⃣ Train/Test split & Model
# ============================

X_train, X_test, y_train, y_test = train_test_split(X_vec, y_enc, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

acc = model.score(X_test, y_test)
print(f"✅ Training completed successfully with accuracy: {acc:.2f}")

# ============================
# 4️⃣ Save model artifacts
# ============================

model_dir = os.path.join(base_dir, "models")
os.makedirs(model_dir, exist_ok=True)

pickle.dump(model, open(os.path.join(model_dir, "model.pkl"), "wb"))
pickle.dump(vectorizer, open(os.path.join(model_dir, "vectorizer.pkl"), "wb"))
pickle.dump(label_encoder, open(os.path.join(model_dir, "label_encoder.pkl"), "wb"))

df.to_csv(os.path.join(model_dir, "training_samples_full.csv"), index=False)

print("🎯 Model and data saved successfully in 'models' folder!")
