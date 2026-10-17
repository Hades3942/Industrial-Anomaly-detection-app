"""
MVTec AD Anomaly Detector - Streamlit prototype (Group 3)

Pages: Home | Data Dashboard | Model Performance | Make Prediction | About

The app loads the model saved in Assignment 4 (models/best_model_*.joblib), rebuilds the
exact same feature vector used during training, and produces a live prediction.
Run with:  streamlit run app.py
"""
import io
import sqlite3
from datetime import datetime
from pathlib import Path

import joblib
import matplotlib
try:
    import cv2
except Exception:
    cv2 = None

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from sklearn.linear_model import LogisticRegression

# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
CSV_PATH = DATA_DIR / "cleaned_project_dataset.csv"
FEATURES_CSV_PATH = DATA_DIR / "model_features.csv"  # optional (exported from Colab)
DB_PATH = DATA_DIR / "predictions.db"

RESIZE_TO = (128, 128)  
MAX_UPLOAD_MB = 10
MIN_IMAGE_SIDE = 32

FEATURE_NAMES = [
    "mean_r", "mean_g", "mean_b", "std_r", "std_g", "std_b",
    "brightness", "contrast", "edge_density",
]

# label, min, max, fallback default, step, help text
FEATURE_SPECS = {
    "mean_r": ("Mean red", 0.0, 255.0, 130.0, 1.0, "Average red value of the image (0-255)."),
    "mean_g": ("Mean green", 0.0, 255.0, 130.0, 1.0, "Average green value of the image (0-255)."),
    "mean_b": ("Mean blue", 0.0, 255.0, 130.0, 1.0, "Average blue value of the image (0-255)."),
    "std_r": ("Std. dev. red", 0.0, 128.0, 60.0, 1.0, "Spread of red values (0-128)."),
    "std_g": ("Std. dev. green", 0.0, 128.0, 60.0, 1.0, "Spread of green values (0-128)."),
    "std_b": ("Std. dev. blue", 0.0, 128.0, 60.0, 1.0, "Spread of blue values (0-128)."),
    "brightness": ("Brightness", 0.0, 255.0, 130.0, 1.0, "Average grayscale intensity (0-255)."),
    "contrast": ("Contrast", 0.0, 128.0, 60.0, 1.0, "Standard deviation of grayscale intensity (0-128)."),
    "edge_density": ("Edge density", 0.0, 1.0, 0.05, 0.005, "Share of pixels detected as edges (0-1)."),
}
SIZE_MIN, SIZE_MAX = MIN_IMAGE_SIDE, 10000

# Results from the submitted google colab  (test set: 652 images, 495 normal / 157 anomalous)
RESULTS = pd.DataFrame(
    {
        "Model": ["Random Forest", "Decision Tree", "Logistic Regression"],
        "Accuracy": [0.852761, 0.714724, 0.601227],
        "Precision (anomaly)": [0.729323, 0.442231, 0.352436],
        "Recall (anomaly)": [0.617834, 0.707006, 0.783439],
        "F1 (anomaly)": [0.668966, 0.544118, 0.486166],
    }
)
# [[true normal->pred normal, true normal->pred anomalous], [true anomalous->pred normal, ->pred anomalous]]
CONFUSION = {
    "Logistic Regression": [[269, 226], [34, 123]],
    "Decision Tree": [[355, 140], [46, 111]],
    "Random Forest": [[459, 36], [60, 97]],
}
TOP_FEATURES = {
    "edge_density": 0.132981, "std_b": 0.119105, "mean_b": 0.117950, "mean_r": 0.112923,
    "mean_g": 0.104675, "brightness": 0.099225, "std_g": 0.095979, "std_r": 0.093235,
    "contrast": 0.090172, "width": 0.005816,
}

st.set_page_config(page_title="MVTec AD Anomaly Detector", page_icon="🔍", layout="wide")


# --------------------------------------------------------------------------------------
# Loading helpers
# --------------------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_artifacts():
    """Load the saved model, feature-column list and (optional) scaler. Never retrains."""
    try:
        model_files = sorted(MODEL_DIR.glob("best_model_*.joblib"))
        if not model_files:
            return {"error": f"No model file (best_model_*.joblib) found in '{MODEL_DIR.name}/'."}
        cols_path = MODEL_DIR / "feature_columns.joblib"
        if not cols_path.exists():
            return {"error": "feature_columns.joblib not found in the models/ folder."}
        model = joblib.load(model_files[0])
        columns = list(joblib.load(cols_path))
        scaler_path = MODEL_DIR / "feature_scaler.joblib"
        scaler = joblib.load(scaler_path) if scaler_path.exists() else None
        name = model_files[0].stem.replace("best_model_", "").replace("_", " ").title()
        return {"model": model, "columns": columns, "scaler": scaler, "name": name}
    except Exception as exc:  # corrupted/incompatible files
        return {"error": f"Could not load the saved model files: {exc}"}


@st.cache_data(show_spinner=False)
def load_dataset():
    if not CSV_PATH.exists():
        return None
    return pd.read_csv(CSV_PATH)


@st.cache_data(show_spinner=False)
def load_feature_table():
    if not FEATURES_CSV_PATH.exists():
        return None
    return pd.read_csv(FEATURES_CSV_PATH)


def get_categories(columns):
    return [c.replace("category_", "", 1) for c in columns if c.startswith("category_")]


# --------------------------------------------------------------------------------------
# Feature extraction / model input 
# --------------------------------------------------------------------------------------
def extract_features(file_bytes):
    """Return (features_dict, width, height, RGB PIL image) from raw image bytes."""
    if len(file_bytes) > MAX_UPLOAD_MB * 1024 * 1024:
        raise ValueError(f"The file is larger than {MAX_UPLOAD_MB} MB.")
    try:
        img = Image.open(io.BytesIO(file_bytes))
        img.load()
        img = img.convert("RGB")
    except Exception:
        raise ValueError("The file could not be read as an image. Please upload a valid PNG or JPG.")
    width, height = img.size
    if min(width, height) < MIN_IMAGE_SIDE:
        raise ValueError(f"The image is too small (minimum {MIN_IMAGE_SIDE}x{MIN_IMAGE_SIDE} pixels).")

    small = img.resize(RESIZE_TO)
    arr = np.array(small)
    if cv2 is not None:
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 100, 200)
    else:
        # Deployment fallback when OpenCV is missing: grayscale + simple gradient threshold.
        gray = (0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]).astype(np.float32)
        gx = np.zeros_like(gray)
        gy = np.zeros_like(gray)
        gx[:, 1:] = np.abs(gray[:, 1:] - gray[:, :-1])
        gy[1:, :] = np.abs(gray[1:, :] - gray[:-1, :])
        grad_mag = np.hypot(gx, gy)
        edges = (grad_mag > 25.0).astype(np.uint8) * 255
    feats = {
        "mean_r": float(arr[:, :, 0].mean()), "mean_g": float(arr[:, :, 1].mean()),
        "mean_b": float(arr[:, :, 2].mean()),
        "std_r": float(arr[:, :, 0].std()), "std_g": float(arr[:, :, 1].std()),
        "std_b": float(arr[:, :, 2].std()),
        "brightness": float(gray.mean()), "contrast": float(gray.std()),
        "edge_density": float(edges.mean() / 255.0),
    }
    return feats, width, height, img


def validate_inputs(values, category, width, height, categories):
    """Return a list of human-readable problems (empty list = inputs are valid)."""
    problems = []
    for name in FEATURE_NAMES:
        label, lo, hi = FEATURE_SPECS[name][0], FEATURE_SPECS[name][1], FEATURE_SPECS[name][2]
        v = values.get(name)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            problems.append(f"{label} is missing.")
        elif not (lo <= v <= hi):
            problems.append(f"{label} must be between {lo:g} and {hi:g} (got {v:g}).")
    for label, v in (("Width", width), ("Height", height)):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            problems.append(f"{label} is missing.")
        elif not (SIZE_MIN <= v <= SIZE_MAX):
            problems.append(f"{label} must be between {SIZE_MIN} and {SIZE_MAX} pixels (got {v:g}).")
    if not category or category not in categories:
        problems.append("Please choose a valid object category.")
    return problems


def build_model_input(values, category, width, height, columns):
    """Arrange inputs in exactly the column order used for training (one-hot category)."""
    row = {c: 0.0 for c in columns}
    row["width"], row["height"] = float(width), float(height)
    row.update({k: float(v) for k, v in values.items()})
    cat_col = f"category_{category}"
    if cat_col not in row:
        raise ValueError(f"Unknown category '{category}'.")
    row[cat_col] = 1.0
    return pd.DataFrame([row], columns=columns)


def predict_anomaly(artifacts, X):
    """Apply the same preprocessing as training and return P(anomalous)."""
    model, scaler = artifacts["model"], artifacts["scaler"]
    X_in = X
    # In Assignment 4 only Logistic Regression was trained on scaled features.
    if isinstance(model, LogisticRegression) and scaler is not None:
        X_in = pd.DataFrame(scaler.transform(X), columns=X.columns)
    proba = model.predict_proba(X_in)[0]
    return float(proba[list(model.classes_).index(1)])


# --------------------------------------------------------------------------------------
# Prediction history (small SQLite log - numeric values only, no images stored)
# --------------------------------------------------------------------------------------
def _db():
    DATA_DIR.mkdir(exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    con.execute(
        "CREATE TABLE IF NOT EXISTS predictions ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT, category TEXT, source TEXT, "
        "edge_density REAL, brightness REAL, contrast REAL, anomaly_probability REAL, result TEXT)"
    )
    return con


def log_prediction(category, source, values, prob, result):
    try:
        con = _db()
        con.execute(
            "INSERT INTO predictions (created_at, category, source, edge_density, brightness, contrast, "
            "anomaly_probability, result) VALUES (?,?,?,?,?,?,?,?)",
            (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), category, source,
             values["edge_density"], values["brightness"], values["contrast"], prob, result),
        )
        con.commit()
        con.close()
        return True
    except Exception:
        return False


def read_history(limit=20):
    try:
        con = _db()
        df = pd.read_sql_query(
            "SELECT created_at AS time, category, source, edge_density, brightness, contrast, "
            "anomaly_probability, result FROM predictions ORDER BY id DESC LIMIT ?", con, params=(limit,))
        con.close()
        return df
    except Exception:
        return pd.DataFrame()


# --------------------------------------------------------------------------------------
# Plot helpers
# --------------------------------------------------------------------------------------
def fig_label_distribution(df):
    counts = df["image_label"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(4.5, 3.4))
    bars = ax.bar(["Normal (0)", "Anomalous (1)"], [counts.get(0, 0), counts.get(1, 0)],
                  color=["#4c9f70", "#d9534f"])
    ax.bar_label(bars)
    ax.set_ylabel("Number of images")
    ax.set_title("Class distribution")
    fig.tight_layout()
    return fig


def fig_category_stacked(df):
    ct = pd.crosstab(df["category"], df["image_label"]).rename(columns={0: "Normal", 1: "Anomalous"})
    for col in ("Normal", "Anomalous"):
        if col not in ct.columns:
            ct[col] = 0
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    ct[["Normal", "Anomalous"]].plot(kind="bar", stacked=True, ax=ax, color=["#4c9f70", "#d9534f"])
    ax.set_xlabel("")
    ax.set_ylabel("Images")
    ax.set_title("Images per category and class")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    return fig, ct


def fig_split_label(df):
    ct = pd.crosstab(df["split"], df["image_label"]).rename(columns={0: "Normal", 1: "Anomalous"})
    for col in ("Normal", "Anomalous"):
        if col not in ct.columns:
            ct[col] = 0
    fig, ax = plt.subplots(figsize=(4.5, 3.4))
    ct[["Normal", "Anomalous"]].plot(kind="bar", ax=ax, color=["#4c9f70", "#d9534f"])
    ax.set_xlabel("Original MVTec split")
    ax.set_ylabel("Images")
    ax.set_title("Original split vs. class")
    plt.setp(ax.get_xticklabels(), rotation=0)
    fig.tight_layout()
    return fig, ct


def fig_edge_density(feat_df):
    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    for label, name, color in ((0, "Normal", "#4c9f70"), (1, "Anomalous", "#d9534f")):
        ax.hist(feat_df.loc[feat_df["image_label"] == label, "edge_density"], bins=30, alpha=0.6,
                label=name, color=color, density=True)
    ax.set_xlabel("edge_density")
    ax.set_ylabel("Density")
    ax.set_title("Edge density by class")
    ax.legend()
    fig.tight_layout()
    return fig


def fig_model_comparison():
    metrics = ["Accuracy", "Precision (anomaly)", "Recall (anomaly)", "F1 (anomaly)"]
    x = np.arange(len(metrics))
    width = 0.26
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    for i, (_, row) in enumerate(RESULTS.iterrows()):
        ax.bar(x + (i - 1) * width, [row[m] for m in metrics], width, label=row["Model"])
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1)
    ax.set_title("Model comparison (test set)")
    ax.legend()
    fig.tight_layout()
    return fig


def fig_confusion(name):
    cm = np.array(CONFUSION[name])
    fig, ax = plt.subplots(figsize=(3.4, 3.1))
    ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center", color="white" if v > cm.max() / 2 else "black")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Normal", "Anomalous"])
    ax.set_yticklabels(["Normal", "Anomalous"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(name, fontsize=10)
    fig.tight_layout()
    return fig


def fig_feature_importance():
    s = pd.Series(TOP_FEATURES).sort_values()
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    ax.barh(s.index, s.values, color="#1f77b4")
    ax.set_xlabel("Importance")
    ax.set_title("Top 10 predictors - Random Forest")
    fig.tight_layout()
    return fig


def show(fig):
    st.pyplot(fig)
    plt.close(fig)


# --------------------------------------------------------------------------------------
# Pages
# --------------------------------------------------------------------------------------
def page_home():
    st.title("🔍 Visual Anomaly Detection for Industrial Components")
    st.caption("Group 3 | SUZA Data Science Laboratory | MVTec Anomaly Detection (MVTec AD) dataset")

    st.header("The problem")
    st.write(
        "High-precision manufacturing lines must find small surface defects - scratches, cracks, "
        "contamination, deformations - on components before final assembly. Manual inspection is "
        "slow and inconsistent, and defective examples are rare, so it is hard to keep a purely "
        "supervised inspection system up to date."
    )
    st.header("Objectives")
    st.markdown(
        "- Turn the image dataset into a clean, analysable table of image-level features.\n"
        "- Train and compare basic Machine Learning models that separate normal from anomalous components.\n"
        "- Deliver the best model as an easy-to-use web application with live predictions.\n"
        "- Be open about what the model can and cannot do, as a baseline for the planned deep-feature pipeline."
    )
    st.header("Proposed solution")
    st.write(
        "For every image the app computes nine simple visual features (colour averages and spreads, "
        "brightness, contrast, edge density). Together with the object category and image size, these "
        "are passed to a **Random Forest** classifier that estimates the probability that the "
        "component is anomalous."
    )
    st.info(
        "**How to use:** open **Make Prediction**, upload an image (or type feature values), choose the object "
        "category and press **Predict**. Explore **Data Dashboard** and **Model Performance** to see how the model was built."
    )


def page_dashboard():
    st.title("📊 Data Dashboard")
    df = load_dataset()
    if df is None:
        st.error(f"Dataset file not found at data/{CSV_PATH.name}.")
        return

    st.header("Dataset summary")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Images (rows)", f"{len(df):,}")
    c2.metric("Object categories", df["category"].nunique())
    share = df["image_label"].mean()
    c3.metric("Normal images", f"{(1 - share) * 100:.1f}%")
    c4.metric("Anomalous images", f"{share * 100:.1f}%")

    st.write("Each row describes one image. Key variables:")
    st.table(pd.DataFrame(
        {
            "Variable": ["image_path", "category", "split", "defect_type", "image_label", "mask_path", "width / height"],
            "Meaning": [
                "Location of the image file (identifier only)",
                "Object or texture type (e.g. bottle, carpet)",
                "Original MVTec partition (train / test)",
                "Type of defect, or 'good' for normal images",
                "Target: 0 = normal, 1 = anomalous",
                "Ground-truth defect mask (NOT_APPLICABLE for normal images)",
                "Image size in pixels",
            ],
        }
    ))
    with st.expander("Preview the first rows of the cleaned dataset"):
        st.dataframe(df.drop(columns=["image_path", "mask_path"], errors="ignore").head(10), width="stretch")

    st.header("Key graphs")
    g1, g2 = st.columns(2)
    with g1:
        show(fig_label_distribution(df))
        st.caption("About three in four images are normal, so accuracy alone would be misleading - "
                   "precision, recall and F1 on the anomalous class are used instead.")
    with g2:
        fig, ct = fig_category_stacked(df)
        show(fig)
        no_anom = [c for c in ct.index if ct.loc[c, "Anomalous"] == 0]
        note = ("Categories with no anomalous images in this dataset: " + ", ".join(no_anom) + "."
                if no_anom else "Every category contains both normal and anomalous images.")
        st.caption("Categories differ in size and in how many anomalous images they contain. " + note)

    g3, g4 = st.columns(2)
    with g3:
        fig, ct2 = fig_split_label(df)
        show(fig)
        st.caption("The original 'train' partition holds only normal images, so a new stratified "
                   "train/test split was created for supervised learning.")
    with g4:
        feat_df = load_feature_table()
        if feat_df is not None and {"edge_density", "image_label"} <= set(feat_df.columns):
            show(fig_edge_density(feat_df))
            st.caption("Anomalous images tend to differ from normal ones in edge density, the strongest predictor.")
        else:
            defects = df[df["defect_type"] != "good"]["defect_type"].value_counts().head(10).sort_values()
            fig, ax = plt.subplots(figsize=(5.5, 3.4))
            ax.barh(defects.index, defects.values, color="#d9534f")
            ax.set_xlabel("Images")
            ax.set_title("Most frequent defect types")
            fig.tight_layout()
            show(fig)
            st.caption("Defect types vary widely across categories (add data/model_features.csv to see feature distributions here).")


def page_performance():
    st.title("📈 Model Performance")
    st.write(
        "Three basic models were trained on a stratified 80/20 split (2,608 training and 652 test images) and "
        "evaluated on the held-out test set (495 normal, 157 anomalous images). All used balanced class weights."
    )
    st.header("Evaluation measures")
    st.markdown(
        "- **Accuracy** - share of all images classified correctly.\n"
        "- **Precision (anomaly)** - of the images flagged anomalous, how many really were.\n"
        "- **Recall (anomaly)** - of the truly anomalous images, how many were found.\n"
        "- **F1 (anomaly)** - balance of precision and recall; used to choose the best model."
    )
    st.header("Model comparison")
    st.dataframe(RESULTS.style.format({c: "{:.3f}" for c in RESULTS.columns[1:]}), width="stretch", hide_index=True)
    show(fig_model_comparison())

    st.header("Confusion matrices")
    cols = st.columns(3)
    for col, name in zip(cols, ["Logistic Regression", "Decision Tree", "Random Forest"]):
        with col:
            show(fig_confusion(name))

    st.header("Selected model: Random Forest")
    best = RESULTS.iloc[0]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy", f"{best['Accuracy']:.3f}")
    m2.metric("Precision (anomaly)", f"{best['Precision (anomaly)']:.3f}")
    m3.metric("Recall (anomaly)", f"{best['Recall (anomaly)']:.3f}")
    m4.metric("F1 (anomaly)", f"{best['F1 (anomaly)']:.3f}")
    st.success(
        "Random Forest was selected because it has the highest F1-score on the anomalous class (0.669) as well "
        "as the highest accuracy and precision. It raises far fewer false alarms (36) than the other models. "
        "Logistic Regression finds more anomalies (higher recall) but flags many normal images by mistake."
    )
    st.header("Important predictors")
    left, right = st.columns([3, 2])
    with left:
        show(fig_feature_importance())
    with right:
        st.write(
            "**edge_density** is the strongest predictor, followed by the blue-channel statistics and the other "
            "colour features. None of the object-category columns are in the top ten, which suggests the model "
            "responds to visual evidence rather than simply to the object type."
        )


def render_result(prob, category):
    is_anomaly = prob >= 0.5
    st.subheader("Prediction result")
    if is_anomaly:
        st.error(f"⚠️ **Likely ANOMALOUS** - a defect is suspected in this {category.replace('_', ' ')}.")
    else:
        st.success(f"✅ **Likely NORMAL** - no defect is indicated for this {category.replace('_', ' ')}.")
    st.metric("Estimated probability of being anomalous", f"{prob * 100:.1f}%")
    st.progress(min(max(prob, 0.0), 1.0))
    if 0.35 <= prob <= 0.65:
        st.warning("The estimate is close to 50%, so the model is unsure. Please inspect this item manually.")
    st.write(
        "**What this means:** the model compares the image's colour, contrast and edge statistics with the "
        "patterns it learned from normal and defective examples. A high probability means the image looks more "
        "like the defective examples. "
        + ("Send this item for manual inspection." if is_anomaly
           else "A normal result is not a guarantee - small defects can be missed.")
    )
    return "Anomalous" if is_anomaly else "Normal"


def page_prediction(artifacts):
    st.title("🧪 Make Prediction")
    if "error" in artifacts:
        st.error(artifacts["error"])
        st.info("Copy best_model_random_forest.joblib, feature_columns.joblib and feature_scaler.joblib "
                "from Assignment 4 into the app's models/ folder, then reload the page.")
        return

    columns, categories = artifacts["columns"], get_categories(artifacts["columns"])
    st.write(f"Model in use: **{artifacts['name']}**. Choose how to provide the input:")
    mode = st.radio("Input method", ["Upload an image", "Enter feature values manually"], horizontal=True,
                    label_visibility="collapsed")

    feat_df = load_feature_table()
    defaults = {}
    for name in FEATURE_NAMES:
        d = FEATURE_SPECS[name][3]
        if feat_df is not None and name in feat_df.columns:
            d = float(np.clip(feat_df[name].median(), FEATURE_SPECS[name][1], FEATURE_SPECS[name][2]))
        defaults[name] = d

    values, width, height, source, ready = None, None, None, None, False
    category = None

    if mode == "Upload an image":
        source = "image upload"
        with st.form("upload_form"):
            category = st.selectbox("Object category", categories, help="The type of component in the image.")
            uploaded = st.file_uploader("Component image (PNG or JPG, max 10 MB)", type=["png", "jpg", "jpeg"])
            submitted = st.form_submit_button("Predict", type="primary")
        if submitted:
            if uploaded is None:
                st.warning("Please upload an image first.")
            else:
                try:
                    values, width, height, img = extract_features(uploaded.getvalue())
                    st.image(img, caption=f"Uploaded image ({width}x{height} px)", width=300)
                    ready = True
                except ValueError as exc:
                    st.error(str(exc))
    else:
        source = "manual entry"
        with st.form("manual_form"):
            category = st.selectbox("Object category", categories, help="The type of component.")
            c1, c2 = st.columns(2)
            width = c1.number_input("Image width (px)", SIZE_MIN, SIZE_MAX, 900, 1)
            height = c2.number_input("Image height (px)", SIZE_MIN, SIZE_MAX, 900, 1)
            values = {}
            cols = st.columns(3)
            for i, name in enumerate(FEATURE_NAMES):
                label, lo, hi, _, step, hint = FEATURE_SPECS[name]
                values[name] = cols[i % 3].number_input(
                    label, min_value=lo, max_value=hi, value=defaults[name], step=step, help=hint,
                    format="%.3f" if name == "edge_density" else "%.1f")
            submitted = st.form_submit_button("Predict", type="primary")
        if submitted:
            problems = validate_inputs(values, category, width, height, categories)
            if problems:
                for p in problems:
                    st.error(p)
            else:
                ready = True

    if ready:
        problems = validate_inputs(values, category, width, height, categories)
        if problems:
            for p in problems:
                st.error(p)
            return
        try:
            X = build_model_input(values, category, width, height, columns)
            prob = predict_anomaly(artifacts, X)
        except Exception as exc:
            st.error(f"The prediction could not be made: {exc}")
            return
        result = render_result(prob, category)
        with st.expander("Values sent to the model"):
            st.dataframe(pd.DataFrame([{**values, "width": width, "height": height, "category": category}]).T
                         .rename(columns={0: "value"}), width="stretch")
        if not log_prediction(category, source, values, prob, result):
            st.caption("Prediction history could not be saved in this environment.")

    hist = read_history()
    if not hist.empty:
        st.subheader("Recent predictions")
        st.dataframe(hist, width="stretch", hide_index=True)
        st.download_button("Download history (CSV)", hist.to_csv(index=False).encode("utf-8"),
                           "prediction_history.csv", "text/csv")


def page_about():
    st.title("ℹ️ About")
    st.header("Intended use")
    st.write(
        "This prototype is an academic demonstration for the SUZA Data Science Laboratory. It shows how a basic "
        "Machine Learning model can give a quick first estimate of whether an industrial component image looks "
        "normal or anomalous, using the MVTec AD benchmark."
    )
    st.header("Key limitations")
    st.markdown(
        "- **Simple features.** The model uses global statistics (colour, brightness, contrast, edge density), not "
        "deep image features, so small or subtle defects can be missed.\n"
        "- **Image-level only.** It says whether an image looks anomalous but does **not** show where the defect is.\n"
        "- **Imperfect accuracy.** On the test set the model found about 62% of anomalous images "
        "(recall 0.618) and about 73% of its alerts were correct (precision 0.729).\n"
        "- **Benchmark data.** It was trained on MVTec AD categories under controlled imaging conditions; "
        "results on other products, cameras or lighting are unknown.\n"
        "- **Image format.** Training used PNG images; heavily compressed JPGs can change the features slightly.\n"
        "- **Imbalanced categories.** Some categories have few or no anomalous examples, so behaviour there is less reliable."
    )
    st.header("Responsible-use statement")
    st.warning(
        "This tool must **not** be used as the only basis for accepting or rejecting real products, or for any "
        "safety-critical decision. Its output is decision support: uncertain or 'anomalous' results should be "
        "checked by a trained human inspector. Uploaded images are processed in memory and are not stored; only "
        "numeric feature values and the result are saved in the local prediction history."
    )
    st.caption("Dataset: MVTec AD (MVTec Software GmbH), licensed CC BY-NC-SA 4.0 - non-commercial use with attribution.")


# --------------------------------------------------------------------------------------
# Navigation
# --------------------------------------------------------------------------------------
def main():
    artifacts = load_artifacts()
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Home", "Data Dashboard", "Model Performance", "Make Prediction", "About"],
                            label_visibility="collapsed")
    st.sidebar.markdown("---")
    if "error" in artifacts:
        st.sidebar.error("Model not loaded")
    else:
        st.sidebar.success(f"Model loaded: {artifacts['name']}")

    if page == "Home":
        page_home()
    elif page == "Data Dashboard":
        page_dashboard()
    elif page == "Model Performance":
        page_performance()
    elif page == "Make Prediction":
        page_prediction(artifacts)
    else:
        page_about()


main()
