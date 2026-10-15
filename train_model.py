from pathlib import Path
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

fake_path = DATA_DIR / "Fake.csv"
real_path = DATA_DIR / "True.csv"

if not fake_path.exists() or not real_path.exists():
    raise FileNotFoundError(
        "Dataset files not found. Download the Kaggle ISOT Fake and Real News Dataset "
        "and place Fake.csv and True.csv in the project's data folder."
    )

fake_df = pd.read_csv(fake_path)
real_df = pd.read_csv(real_path)

fake_df.columns = fake_df.columns.str.strip().str.lower()
real_df.columns = real_df.columns.str.strip().str.lower()
fake_df["label"] = "FAKE"
real_df["label"] = "REAL"
df = pd.concat([fake_df, real_df], ignore_index=True)

for column in ["title", "text"]:
    if column not in df.columns:
        df[column] = ""
    df[column] = df[column].fillna("").astype(str)

df["content"] = (df["title"].str.strip() + " " + df["text"].str.strip()).str.strip()
df = df[df["content"].str.len() > 0].copy()
df = df.drop_duplicates(subset=["content"])
df = df[["content", "label"]].sample(frac=1, random_state=42).reset_index(drop=True)

X = df["content"]
y = df["label"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

model = Pipeline([
    ("tfidf", TfidfVectorizer(
        stop_words="english",
        max_features=100000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True
    )),
    ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced"))
])

print(f"Articles after preprocessing: {len(df):,}")
print(f"Training samples: {len(X_train):,}")
print(f"Testing samples: {len(X_test):,}")
print("Training model... this may take a few minutes.")
model.fit(X_train, y_train)

pred = model.predict(X_test)
metrics = {
    "Accuracy": accuracy_score(y_test, pred),
    "Precision (FAKE)": precision_score(y_test, pred, pos_label="FAKE", zero_division=0),
    "Recall (FAKE)": recall_score(y_test, pred, pos_label="FAKE", zero_division=0),
    "F1-score (FAKE)": f1_score(y_test, pred, pos_label="FAKE", zero_division=0)
}

joblib.dump(model, MODEL_DIR / "fake_news_model.pkl")
pd.DataFrame([metrics]).to_csv(MODEL_DIR / "model_evaluation_metrics.csv", index=False)
pd.DataFrame({
    "News Article": X_test.reset_index(drop=True),
    "Actual Label": y_test.reset_index(drop=True),
    "Predicted Label": pred
}).to_csv(MODEL_DIR / "test_predictions.csv", index=False)

print("\nMODEL EVALUATION RESULTS")
for name, score in metrics.items():
    print(f"{name}: {score:.4f} ({score:.2%})")
print("\nClassification report:")
print(classification_report(y_test, pred, labels=["FAKE", "REAL"], zero_division=0))
print("\nConfusion matrix (rows=actual, columns=predicted; order FAKE, REAL):")
print(confusion_matrix(y_test, pred, labels=["FAKE", "REAL"]))
print(f"\nSaved model to: {MODEL_DIR / 'fake_news_model.pkl'}")
print("Training complete. Start the app with: streamlit run app.py")
