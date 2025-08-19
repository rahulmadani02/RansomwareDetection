import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

df = pd.read_csv("opcode_dataset_large.csv")
X_train, X_test, y_train, y_test = train_test_split(df['opcode_sequence'], df['label'], test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train_vec, y_train)

joblib.dump(model, "model.joblib")
joblib.dump(vectorizer, "vectorizer.joblib")
print("✅ Model and vectorizer saved.")
