"""Educational baseline for prioritizing ClinVar variant records.
Not for clinical use. It downloads public ClinVar summary data at runtime.
"""
import io
import gzip
import re
import urllib.request

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.model_selection import GroupShuffleSplit
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report,
)

st.set_page_config(page_title="Genetic Mutation Prioritization", layout="wide")
st.title("AI-Based Prioritization of Genetic Mutations")
st.write(
    "Educational machine-learning prototype using public NCBI ClinVar records. "
    "It learns to distinguish ClinVar-labelled Benign/Likely benign variants from "
    "Pathogenic/Likely pathogenic variants. It is not a medical diagnostic tool."
)
st.info(
    "The model uses basic features extracted from ClinVar records, including variant type, "
    "allele lengths, chromosome, and simple HGVS-derived features. It does not use a full "
    "protein-structure or functional-impact analysis."
)

CLINVAR_URL = "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/variant_summary.txt.gz"

@st.cache_data(show_spinner=False)
def load_clinvar():
    request = urllib.request.Request(
        CLINVAR_URL, headers={"User-Agent": "GeneticMutationStudentProject/1.1"}
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        compressed = response.read()
    with gzip.GzipFile(fileobj=io.BytesIO(compressed)) as gz:
        return pd.read_csv(gz, sep="\t", low_memory=False, dtype=str)


def first_existing(df, names):
    return next((name for name in names if name in df.columns), None)


def infer_consequence(name, variant_type):
    """Infer a rough consequence category from ClinVar's variant name/HGVS text."""
    text = str(name or "").lower()
    typ = str(variant_type or "").lower()
    protein = re.search(r"p\.([^\s,)]+)", text)
    ptext = protein.group(1).lower() if protein else ""

    if "fs" in ptext or "frameshift" in text:
        return "frameshift"
    if "ter" in ptext or "stop_gained" in text or re.search(r"p\.[a-z]{1,3}\d+\*", text):
        return "stop_gained"
    if "splice" in text or re.search(r"c\.[-*]?\d+[+-]\d+", text):
        return "splice_related"
    if re.fullmatch(r"[a-z]{1,3}\d+[a-z]{1,3}", ptext):
        return "missense_or_amino_acid_change"
    if "=" in ptext or re.search(r"c\.[-*]?\d+[a-z]>[a-z].*p\.[^ ]+=", text):
        return "synonymous_or_no_protein_change"
    if "deletion" in typ or "deletion" in text or "del" in typ:
        return "deletion"
    if "duplication" in typ or "duplication" in text or "dup" in typ:
        return "duplication"
    if "insertion" in typ or "insertion" in text or "ins" in typ:
        return "insertion"
    if "copy number" in typ or "cnv" in typ:
        return "copy_number_change"
    if "single nucleotide" in typ or "snv" in typ or "single nucleotide" in text:
        return "single_nucleotide_other"
    return "other_or_unknown"


def extract_number(pattern, text):
    match = re.search(pattern, str(text or ""), flags=re.IGNORECASE)
    return float(match.group(1)) if match else np.nan


def prepare_gene_data(raw, gene):
    gene_col = first_existing(raw, ["GeneSymbol", "Gene"])
    sig_col = first_existing(raw, ["ClinicalSignificance", "Clinical significance"])
    if gene_col is None or sig_col is None:
        raise ValueError(
            "Could not find the gene or clinical-significance columns in the downloaded file. "
            f"Columns found include: {', '.join(raw.columns[:35])}"
        )

    df = raw.copy()
    df = df[df[gene_col].fillna("").str.split(r"[,;|]").apply(
        lambda values: gene.upper() in [value.strip().upper() for value in values]
    )].copy()
    if df.empty:
        raise ValueError(f"No ClinVar records were found for {gene} in the downloaded file.")

    # Prefer GRCh38 so coordinates are more consistent; fall back if unavailable.
    assembly_col = first_existing(df, ["Assembly"])
    if assembly_col:
        grch38 = df[assembly_col].fillna("").str.contains("GRCh38", case=False)
        if grch38.any():
            df = df[grch38].copy()

    sig = df[sig_col].fillna("").str.lower()
    excluded = sig.str.contains(
        r"conflict|uncertain|risk factor|association|protective|drug response|not provided|other",
        regex=True,
    )
    has_benign = sig.str.contains("benign", regex=False)
    has_pathogenic = sig.str.contains("pathogenic", regex=False)
    keep_benign = has_benign & ~has_pathogenic & ~excluded
    keep_pathogenic = has_pathogenic & ~has_benign & ~excluded
    df = df[keep_benign | keep_pathogenic].copy()
    if df.empty:
        raise ValueError(
            f"No unambiguous Benign/Pathogenic records remained for {gene}. Try another gene."
        )
    sig = df[sig_col].fillna("").str.lower()
    df["Label"] = np.where(sig.str.contains("pathogenic", regex=False), "Pathogenic", "Benign")

    # If available, keep one record per ClinVar VariationID to reduce duplicate rows.
    variation_id_col = first_existing(df, ["VariationID"])
    if variation_id_col:
        df = df.drop_duplicates(subset=[variation_id_col], keep="first").copy()

    pos_col = first_existing(df, ["PositionVCF", "Start", "Position"])
    ref_col = first_existing(df, ["ReferenceAlleleVCF", "ReferenceAllele", "Reference allele"])
    alt_col = first_existing(df, ["AlternateAlleleVCF", "AlternateAllele", "Alternate allele"])
    type_col = first_existing(df, ["Type", "VariantType"])
    name_col = first_existing(df, ["Name"])
    chromosome_col = first_existing(df, ["Chromosome"])

    df["position"] = pd.to_numeric(df[pos_col], errors="coerce") if pos_col else np.nan
    df["ref_length"] = df[ref_col].fillna("").str.len().astype(float) if ref_col else 0.0
    df["alt_length"] = df[alt_col].fillna("").str.len().astype(float) if alt_col else 0.0
    df["variant_type"] = df[type_col].fillna("Unknown").astype(str) if type_col else "Unknown"
    df["chromosome"] = df[chromosome_col].fillna("Unknown").astype(str) if chromosome_col else "Unknown"
    df["variant_name"] = df[name_col].fillna("").astype(str) if name_col else ""
    df["consequence_group"] = [infer_consequence(n, t) for n, t in zip(df["variant_name"], df["variant_type"])]
    df["protein_position"] = df["variant_name"].apply(
        lambda value: extract_number(r"p\.(?:[A-Za-z]{1,3})(\d+)", value)
    )
    df["cdna_position"] = df["variant_name"].apply(
        lambda value: extract_number(r"c\.[-*]?(\d+)", value)
    )

    valid_positions = df["position"].dropna()
    if len(valid_positions) > 1 and valid_positions.max() > valid_positions.min():
        low, high = valid_positions.min(), valid_positions.max()
        df["relative_gene_position"] = (df["position"] - low) / (high - low)
    else:
        df["relative_gene_position"] = 0.5

    # Group by protein position where available; otherwise genomic position.
    # This makes the holdout test less likely to contain the same site in both sets.
    df["evaluation_group"] = np.where(
        df["protein_position"].notna(),
        "protein_" + df["protein_position"].fillna(-1).astype(str),
        "genomic_" + df["position"].fillna(-1).astype(str),
    )

    features = [
        "chromosome", "variant_type", "consequence_group", "ref_length", "alt_length",
        "relative_gene_position", "protein_position", "cdna_position",
    ]
    model_df = df[features + ["Label", "evaluation_group"]].replace([np.inf, -np.inf], np.nan)
    model_df = model_df.dropna(subset=["Label"])
    model_df = model_df.dropna(subset=["relative_gene_position"], how="all")
    return df, model_df


def make_pipeline():
    numeric_features = [
        "ref_length", "alt_length", "relative_gene_position", "protein_position", "cdna_position"
    ]
    categorical_features = ["chromosome", "variant_type", "consequence_group"]
    numeric_pipe = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipe, numeric_features),
        ("categorical", categorical_pipe, categorical_features),
    ])
    return Pipeline([
        ("preprocess", preprocessor),
        ("model", RandomForestClassifier(
            n_estimators=300, random_state=42, class_weight="balanced", min_samples_leaf=2,
            n_jobs=-1,
        )),
    ])


def grouped_train_test_split(model_df):
    X = model_df.drop(columns=["Label", "evaluation_group"])
    y = model_df["Label"].reset_index(drop=True)
    groups = model_df["evaluation_group"].reset_index(drop=True)
    if y.nunique() < 2:
        raise ValueError("The selected gene needs both Benign and Pathogenic examples.")
    counts = y.value_counts()
    if counts.min() < 4:
        raise ValueError("There are too few records in one class for a useful grouped evaluation. Try BRCA1, TP53, or CFTR.")

    # Try several grouped splits and select one containing both labels in train and test.
    splitter = GroupShuffleSplit(n_splits=60, test_size=0.25, random_state=42)
    selected = None
    for train_idx, test_idx in splitter.split(X, y, groups):
        if y.iloc[train_idx].nunique() == 2 and y.iloc[test_idx].nunique() == 2:
            selected = train_idx, test_idx
            break
    if selected is None:
        raise ValueError(
            "Could not make a grouped train/test split with both classes. "
            "Try another gene or collect more labelled variants."
        )
    train_idx, test_idx = selected
    return X.iloc[train_idx], X.iloc[test_idx], y.iloc[train_idx], y.iloc[test_idx]


def train_model(model_df):
    X_train, X_test, y_train, y_test = grouped_train_test_split(model_df)
    clf = make_pipeline()
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    proba = clf.predict_proba(X_test)
    classes = list(clf.named_steps["model"].classes_)
    path_idx = classes.index("Pathogenic")
    try:
        auc = roc_auc_score((y_test == "Pathogenic").astype(int), proba[:, path_idx])
    except ValueError:
        auc = np.nan
    metrics = {
        "Accuracy": accuracy_score(y_test, pred),
        "Balanced accuracy": balanced_accuracy_score(y_test, pred),
        "Precision (Pathogenic)": precision_score(y_test, pred, pos_label="Pathogenic", zero_division=0),
        "Recall (Pathogenic)": recall_score(y_test, pred, pos_label="Pathogenic", zero_division=0),
        "F1 (Pathogenic)": f1_score(y_test, pred, pos_label="Pathogenic", zero_division=0),
        "ROC-AUC": auc,
    }
    return clf, X_test, y_test, pred, metrics


with st.sidebar:
    st.header("Settings")
    gene = st.selectbox("Choose gene", ["BRCA1", "TP53", "CFTR", "APC", "MTHFR", "PMM2"])
    st.caption("A selected gene needs enough unambiguous Benign and Pathogenic records.")
    load_button = st.button("Load data and train model", type="primary")

if "raw_clinvar" not in st.session_state:
    st.session_state.raw_clinvar = None
if "loaded_gene" not in st.session_state:
    st.session_state.loaded_gene = None

if load_button:
    st.session_state.raw_clinvar = None
    st.session_state.loaded_gene = None

if load_button or st.session_state.raw_clinvar is not None:
    try:
        with st.spinner("Downloading ClinVar data. The file is large, so this may take a few minutes..."):
            if st.session_state.raw_clinvar is None:
                st.session_state.raw_clinvar = load_clinvar()
            raw = st.session_state.raw_clinvar
        st.success(f"Downloaded {len(raw):,} ClinVar records.")
        with st.spinner(f"Preparing {gene} data and training the Random Forest model..."):
            original_df, model_df = prepare_gene_data(raw, gene)
            clf, X_test, y_test, pred, metrics = train_model(model_df)
        st.session_state.loaded_gene = gene
        st.session_state.model = clf
        st.session_state.model_df = model_df
        st.session_state.original_df = original_df
        st.session_state.metrics = metrics

        st.subheader(f"Dataset summary: {gene}")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Usable variants", f"{len(model_df):,}")
        c2.metric("Benign", f"{(model_df['Label'] == 'Benign').sum():,}")
        c3.metric("Pathogenic", f"{(model_df['Label'] == 'Pathogenic').sum():,}")
        c4.metric("Test records", f"{len(y_test):,}")

        st.subheader("Model evaluation on held-out groups")
        metric_cols = st.columns(len(metrics))
        for col, (name, value) in zip(metric_cols, metrics.items()):
            col.metric(name, "N/A" if pd.isna(value) else f"{value:.3f}")
        st.caption(
            "The test set is split by protein position where available, or genomic position otherwise. "
            "This is more cautious than a plain random split, but it is still not a clinical validation."
        )

        st.subheader("Confusion matrix")
        labels = ["Benign", "Pathogenic"]
        cm = confusion_matrix(y_test, pred, labels=labels)
        fig, ax = plt.subplots()
        ax.imshow(cm)
        ax.set_xticks([0, 1], labels=labels)
        ax.set_yticks([0, 1], labels=labels)
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("Actual label")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center")
        st.pyplot(fig)
        plt.close(fig)

        st.subheader("Classification report")
        report = classification_report(y_test, pred, labels=labels, zero_division=0, output_dict=True)
        st.dataframe(pd.DataFrame(report).transpose().round(3), use_container_width=True)

        st.subheader("Variant records used")
        display_cols = [c for c in [
            "VariationID", "GeneSymbol", "Name", "Type", "ClinicalSignificance", "ReviewStatus",
            "Chromosome", "position", "ref_length", "alt_length", "consequence_group",
            "protein_position", "cdna_position", "Label",
        ] if c in original_df.columns]
        st.dataframe(original_df[display_cols].head(500), use_container_width=True)
        st.caption("The table shows up to 500 records. Download the full processed dataset below.")
        download_df = model_df.drop(columns=["evaluation_group"])
        st.download_button(
            "Download processed variant dataset (CSV)",
            data=download_df.to_csv(index=False).encode("utf-8"),
            file_name=f"{gene.lower()}_clinvar_processed.csv",
            mime="text/csv",
        )

        st.subheader("Try a sample variant")
        st.write("Enter the available features for a demonstration prediction. This is not a clinical interpretation.")
        chromosome_options = sorted(model_df["chromosome"].dropna().astype(str).unique().tolist())
        type_options = sorted(model_df["variant_type"].dropna().astype(str).unique().tolist())
        consequence_options = sorted(model_df["consequence_group"].dropna().astype(str).unique().tolist())
        with st.form("prediction_form"):
            p1, p2, p3 = st.columns(3)
            chromosome = p1.selectbox("Chromosome", chromosome_options or ["Unknown"])
            vtype = p2.selectbox("Variant type", type_options or ["Unknown"])
            consequence = p3.selectbox("Estimated consequence category", consequence_options or ["other_or_unknown"])
            p4, p5, p6 = st.columns(3)
            ref_len = p4.number_input("Reference allele length", min_value=0, max_value=10000, value=1, step=1)
            alt_len = p5.number_input("Alternate allele length", min_value=0, max_value=10000, value=1, step=1)
            rel_pos = p6.number_input("Relative gene position (0 to 1)", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
            p7, p8 = st.columns(2)
            protein_pos = p7.number_input("Protein position (0 if unknown)", min_value=0, max_value=100000, value=0, step=1)
            cdna_pos = p8.number_input("cDNA position (0 if unknown)", min_value=0, max_value=10000000, value=0, step=1)
            submit = st.form_submit_button("Predict label")
        if submit:
            sample = pd.DataFrame([{
                "chromosome": chromosome,
                "variant_type": vtype,
                "consequence_group": consequence,
                "ref_length": float(ref_len),
                "alt_length": float(alt_len),
                "relative_gene_position": float(rel_pos),
                "protein_position": np.nan if protein_pos == 0 else float(protein_pos),
                "cdna_position": np.nan if cdna_pos == 0 else float(cdna_pos),
            }])
            predicted = clf.predict(sample)[0]
            probabilities = clf.predict_proba(sample)[0]
            classes = list(clf.named_steps["model"].classes_)
            scores = ", ".join(f"{label}: {prob:.1%}" for label, prob in zip(classes, probabilities))
            st.info(f"Model output: **{predicted}**\n\nModel scores: {scores}")
            st.caption("These scores are not clinically meaningful probabilities and must not guide healthcare decisions.")

    except Exception as exc:
        st.error(f"The model could not finish: {exc}")
        st.info(
            "Check your internet connection. Try BRCA1, TP53, or CFTR if the selected gene has too few records. "
            "If the public ClinVar file has changed format, the data-column mapping may need updating."
        )
else:
    st.info("Choose a gene in the sidebar and click **Load data and train model**.")
    st.markdown("""
    ### Run on Windows
    1. Install Python 3.10 or newer.
    2. Open Command Prompt in this folder.
    3. Run `py -m pip install -r requirements.txt`
    4. Run `py -m streamlit run app.py`
    5. The app opens in your browser. Internet is required to download ClinVar data.
    """)
