# app_super_interactive.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import io
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# optional imports for a gauge
import plotly.graph_objects as go

st.set_page_config(page_title="Ransomware Detection", layout="wide")
st.title("🔐 Ransomware Detection")

# Paths and constants
MODEL_FILE = "model.joblib"
VECT_FILE = "vectorizer.joblib"
DATA_FILE = "opcode_dataset_large.csv"
# Utility functions
@st.cache_resource
def load_model_vectorizer():
    m, v = None, None
    try:
        if os.path.exists(MODEL_FILE) and os.path.exists(VECT_FILE):
            m = joblib.load(MODEL_FILE)
            v = joblib.load(VECT_FILE)
    except Exception as e:
        st.warning(f"Could not load model/vectorizer: {e}")
    return m, v

def clean_sequence(s: str) -> str:
    if not isinstance(s, str):
        return ""
    # basic cleaning: lowercase, strip, replace non-alphanum (keep spaces)
    s = s.strip().lower()
    s = " ".join(s.split())  # collapse whitespace
    # keep letters/numbers and spaces and comma? opcodes are letters so remove others
    allowed = "abcdefghijklmnopqrstuvwxyz0123456789 "
    s = "".join(ch for ch in s if ch in allowed)
    return s

def predict_one(model, vect, seq):
    seq_clean = clean_sequence(seq)
    X = vect.transform([seq_clean])
    pred = model.predict(X)[0]
    prob = model.predict_proba(X)[0][pred]
    return pred, float(prob), seq_clean

def predict_batch_df(model, vect, df):
    seqs = df['opcode_sequence'].astype(str).apply(clean_sequence).tolist()
    X = vect.transform(seqs)
    preds = model.predict(X)
    probs = model.predict_proba(X).max(axis=1)
    df2 = df.copy()
    df2['opcode_sequence_clean'] = seqs
    df2['prediction'] = ["Ransomware" if p==1 else "Benign" for p in preds]
    df2['confidence'] = np.round(probs, 3)
    return df2

def plot_gauge(value, title="Confidence"):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value*100,
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={'axis': {'range': [0, 100]},
               'bar': {'color': "darkred" if value>0.5 else "darkgreen"}},
        title={'text': title}
    ))
    st.plotly_chart(fig, use_container_width=True)

# -----------------------
# Load resources (if available)
# -----------------------
model, vect = load_model_vectorizer()
if os.path.exists(DATA_FILE):
    df_full = pd.read_csv(DATA_FILE)
else:
    df_full = None

# -----------------------
# Tabs
# -----------------------
tabs = st.tabs([
    "🧍 Single Prediction",
    "📁 Batch Upload (drag & drop)",
    "📊 Model Metrics",
    "📂 Dataset Preview & Search",
    "📈 Data Visualization (Wordcloud + Charts)",
    "⚙️ Model Training & Tuning",
    "🔁 Model Comparison"
])

# -----------------------
# Tab 1: Single Prediction
# -----------------------
with tabs[0]:
    st.header("Single Prediction (live cleaning + confidence gauge)")
    col1, col2 = st.columns([2,1])
    with col1:
        user_seq = st.text_area("Enter opcode sequence (space-separated)", height=140)
        if st.checkbox("Auto-clean sequence (lowercase, remove invalid chars)", value=True):
            cleaned = clean_sequence(user_seq)
            st.caption("Cleaned sequence:")
            st.code(cleaned)
        if st.button("Predict Single"):
            if model is None or vect is None:
                st.error("Model or vectorizer not loaded. Train model or upload model/vectorizer files.")
            elif not user_seq.strip():
                st.warning("Please enter a sequence.")
            else:
                pred, prob, cleaned = predict_one(model, vect, user_seq)
                label = "🔥 Ransomware" if pred==1 else "✅ Benign"
                st.success(f"Prediction: **{label}** (confidence {prob:.2f})")
                plot_gauge(prob, title="Prediction Confidence")
    with col2:
        st.subheader("Upload model/vectorizer")
        uploaded = st.file_uploader("Upload .joblib files (model then vectorizer) — optional", type=["joblib"], accept_multiple_files=True)
        if uploaded:
            for f in uploaded:
                content = f.read()
                # heuristics: if filename contains 'vector' -> vectorizer
                fname = f.name.lower()
                if "vector" in fname:
                    with open(VECT_FILE, "wb") as vf: vf.write(content)
                else:
                    with open(MODEL_FILE, "wb") as mf: mf.write(content)
            st.success("Saved uploaded files. Reloading model...")
            model, vect = load_model_vectorizer()

# -----------------------
# Tab 2: Batch Upload
# -----------------------
with tabs[1]:
    st.header("Batch Upload & Predictions (drag-and-drop supported)")
    uploaded_file = st.file_uploader("Upload CSV with column 'opcode_sequence'", type="csv")
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            if 'opcode_sequence' not in df.columns:
                st.error("CSV must include 'opcode_sequence' column.")
            else:
                if model is None or vect is None:
                    st.error("Model not loaded. Train or upload model first.")
                else:
                    df_preds = predict_batch_df(model, vect, df)
                    st.success(f"Predicted {len(df_preds)} rows.")
                    st.dataframe(df_preds.head(200))
                    out_csv = df_preds.to_csv(index=False).encode('utf-8')
                    st.download_button("Download predictions (.csv)", out_csv, "predictions.csv")
                    # Provide quick search in results
                    if st.checkbox("Search in results for opcode substring"):
                        q = st.text_input("Enter substring")
                        if q:
                            matches = df_preds[df_preds['opcode_sequence_clean'].str.contains(q)]
                            st.write(f"Found {len(matches)} matches")
                            st.dataframe(matches.head(200))
        except Exception as e:
            st.error(f"Failed to read CSV: {e}")

# Tab 3: Model Metrics
with tabs[2]:
    st.header("Model Metrics & Misclassification Explorer")
    if df_full is None:
        st.warning("Dataset file missing. Place 'opcode_dataset_large.csv' in app folder.")
    elif model is None or vect is None:
        st.error("Model or vectorizer not loaded.")
    else:
        use_sample = st.checkbox("Use sample of dataset for faster metrics", value=True)
        if use_sample:
            sample_n = min(10000, len(df_full))
            sample_df = df_full.sample(n=sample_n, random_state=42)
        else:
            sample_df = df_full
        X = vect.transform(sample_df['opcode_sequence'].astype(str).tolist())
        y_true = sample_df['label'].values
        y_pred = model.predict(X)
        acc = accuracy_score(y_true, y_pred)
        st.metric("Accuracy (on dataset/sample)", f"{acc:.4f}")
        cr = classification_report(y_true, y_pred, output_dict=True)
        st.subheader("Classification Report")
        st.dataframe(pd.DataFrame(cr).transpose().round(3))
        cm = confusion_matrix(y_true, y_pred)
        st.subheader("Confusion Matrix")
        fig, ax = plt.subplots(figsize=(4,3))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)
        # Misclassified explorer
        st.subheader("Misclassified examples")
        idxs = np.where(y_true != y_pred)[0]
        st.write(f"Total misclassified rows in sample: {len(idxs)}")
        if len(idxs) > 0:
            sel = st.number_input("Show first N misclassified", min_value=1, max_value=min(500, len(idxs)), value=10)
            mis = sample_df.iloc[idxs][:sel].copy()
            mis['predicted'] = y_pred[idxs][:sel]
            st.dataframe(mis[['opcode_sequence', 'predicted']])

# Tab 4: Dataset Preview & Search
with tabs[3]:
    st.header("Dataset Preview & Search")
    if df_full is None:
        st.warning("Dataset missing.")
    else:
        st.write(f"Total rows: {len(df_full)}")
        q = st.text_input("Search for opcode substring (e.g., 'jmp')")
        label_filter = st.selectbox("Filter label", options=["All", "Benign (0)", "Ransomware (1)"])
        if label_filter == "All":
            view_df = df_full
        elif label_filter == "Benign (0)":
            view_df = df_full[df_full['label']==0]
        else:
            view_df = df_full[df_full['label']==1]
        if q:
            view_df = view_df[view_df['opcode_sequence'].str.contains(q, na=False)]
            st.write(f"Found {len(view_df)} rows matching '{q}'")
        st.dataframe(view_df.sample(n=min(200, len(view_df))))

# Tab 5: Data Visualization

with tabs[4]:
    st.header("Data Visualization")
    if df_full is None:
        st.warning("Dataset missing.")
    else:
        col1, col2 = st.columns([1,1])
        with col1:
            st.subheader("Label distribution")
            counts = df_full['label'].value_counts().sort_index()
            fig1, ax1 = plt.subplots()
            ax1.pie(counts.values, labels=["Benign","Ransomware"], autopct="%1.1f%%", startangle=90)
            ax1.axis('equal')
            st.pyplot(fig1)
        with col2:
            st.subheader("Top opcodes (word count)")
            tokens = df_full['opcode_sequence'].str.split().explode()
            top = tokens.value_counts().head(20)
            fig2, ax2 = plt.subplots(figsize=(6,4))
            sns.barplot(x=top.values, y=top.index, ax=ax2)
            st.pyplot(fig2)
        st.subheader("Opcode Word Cloud")
        wc_text = " ".join(df_full['opcode_sequence'].astype(str).tolist())
        wc = WordCloud(width=800, height=400, background_color="white").generate(wc_text)
        fig3, ax3 = plt.subplots(figsize=(12,4))
        ax3.imshow(wc, interpolation='bilinear')
        ax3.axis('off')
        st.pyplot(fig3)

# Tab 6: Model Training & Tuning
with tabs[5]:
    st.header("Train New Model (Interactive)")
    if df_full is None:
        st.warning("Dataset missing. Place 'opcode_dataset_large.csv' in the app folder.")
    else:
        st.info("Training will overwrite model.joblib and vectorizer.joblib in app folder.")
        n_estimators = st.slider("n_estimators", 10, 500, 100, step=10)
        test_size = st.slider("test_size (fraction)", 0.05, 0.5, 0.2, 0.05)
        max_features = st.selectbox("max_features for TF-IDF (None or 'sqrt')", options=["None","sqrt"])
        if st.button("Train Model Now"):
            with st.spinner("Training... this may take some time depending on dataset size"):
                try:
                    vect = TfidfVectorizer(max_features=None if max_features=="None" else "sqrt")
                    X = vect.fit_transform(df_full['opcode_sequence'].astype(str).tolist())
                    X_tr, X_te, y_tr, y_te = train_test_split(X, df_full['label'].values, test_size=test_size, stratify=df_full['label'], random_state=42)
                    model_new = RandomForestClassifier(n_estimators=int(n_estimators), n_jobs=-1, random_state=42)
                    model_new.fit(X_tr, y_tr)
                    joblib.dump(model_new, MODEL_FILE)
                    joblib.dump(vect, VECT_FILE)
                    acc = model_new.score(X_te, y_te)
                    st.success(f"Training complete. Test accuracy: {acc:.3f}")
                except Exception as e:
                    st.error(f"Training failed: {e}")

# Tab 7: Model Comparison
with tabs[6]:
    st.header("Model Comparison (Compare two models side-by-side)")
    st.write("Upload two model+vectorizer pairs to compare metrics on the dataset/sample.")
    files = st.file_uploader("Upload up to 4 files (model1, vect1, model2, vect2)", type=["joblib"], accept_multiple_files=True)
    if files and df_full is not None:
        # naive grouping: first two -> model A, next two -> model B
        try:
            # save temporarily
            tmp_dir = "tmp_models"
            os.makedirs(tmp_dir, exist_ok=True)
            for i, f in enumerate(files):
                with open(os.path.join(tmp_dir, f.name), "wb") as out:
                    out.write(f.read())
            # find pairs
            file_names = os.listdir(tmp_dir)
            file_names.sort()
            # try to pair by order
            if len(file_names) < 2:
                st.error("Upload at least two files (model and vectorizer for one model).")
            else:
                def eval_pair(model_path, vect_path, sample_df):
                    m = joblib.load(model_path)
                    v = joblib.load(vect_path)
                    X = v.transform(sample_df['opcode_sequence'].astype(str).tolist())
                    y = sample_df['label'].values
                    y_pred = m.predict(X)
                    return accuracy_score(y, y_pred), classification_report(y, y_pred, output_dict=True)
                # pick first pair
                m1 = os.path.join(tmp_dir, file_names[0])
                v1 = os.path.join(tmp_dir, file_names[1]) if len(file_names)>1 else None
                acc1, rep1 = None, None
                if v1:
                    acc1, rep1 = eval_pair(m1, v1, df_full.sample(n=min(5000,len(df_full)), random_state=1))
                m2, v2 = (None, None)
                acc2, rep2 = (None, None)
                if len(file_names) >= 4:
                    m2 = os.path.join(tmp_dir, file_names[2])
                    v2 = os.path.join(tmp_dir, file_names[3])
                    acc2, rep2 = eval_pair(m2, v2, df_full.sample(n=min(5000,len(df_full)), random_state=2))
                # display
                st.write("Model A accuracy:", acc1)
                if acc2 is not None:
                    st.write("Model B accuracy:", acc2)
                # cleanup
                for f in file_names:
                    os.remove(os.path.join(tmp_dir, f))
                os.rmdir(tmp_dir)
        except Exception as e:
            st.error(f"Comparison failed: {e}")
