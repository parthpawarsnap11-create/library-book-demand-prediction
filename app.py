import streamlit as st
import joblib
import numpy as np
import os
import json
import pandas as pd

# ── CONFIG ────────────────────────────────────────────────────
st.set_page_config(
    page_title="LibraIQ | Library Intelligence Analytics",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model = joblib.load(os.path.join(BASE_DIR, "models", "best_model.pkl"))
METRICS_PATH = os.path.join(BASE_DIR, "models", "model_metrics.json")

SCHOOLS = {
    "VSST (Science & Technology)": ["AI/ML", "Python", "Data Structures", "Calculus", "Database Systems"],
    "Law":    ["Constitutional Law", "Criminal Law", "Corporate Law", "IPR", "Contract Law"],
    "Design": ["UI/UX Design", "Product Design", "Typography", "Graphic Design", "Interaction Design"],
    "MBA":    ["Marketing", "Finance", "Human Resources", "Operations", "Entrepreneurship"],
    "TSM (True School of Music)": ["Guitar", "Piano", "Music Theory", "Vocal Training", "Music Production"],
}
ALL_SUBJECTS = sum(SCHOOLS.values(), [])
ENC = {s: i for i, s in enumerate(ALL_SUBJECTS)}
LABELS = {0: "Low", 1: "Medium", 2: "High"}


def predict_all(semester=4, borrow=25, relevance=1):
    rows = []
    for school, subjects in SCHOOLS.items():
        for s in subjects:
            code = int(model.predict(np.array([[ENC[s], semester, borrow, relevance]]))[0])
            rows.append({"School": school, "Subject": s, "Demand": LABELS[code], "Code": code})
    return pd.DataFrame(rows)


# ── CSS ───────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

* { font-family: 'Inter', sans-serif !important; }
.stApp { background: #f0f4fb !important; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #0b0f1a !important;
    border-right: 1px solid #1a2332 !important;
}
section[data-testid="stSidebar"] > div { padding-top: 0 !important; }

.sb-logo { font-size:21px;font-weight:800;color:#fff;letter-spacing:3px;padding:28px 20px 2px; }
.sb-tag  { font-size:8px;color:#2d4a6a;letter-spacing:2px;text-transform:uppercase;padding:0 20px 20px; }
.sb-sec  { font-size:9px;color:#2d4a6a;font-weight:700;letter-spacing:2px;text-transform:uppercase;padding:14px 20px 6px; }
.sb-hr   { border:none;border-top:1px solid #1a2332;margin:4px 0; }
.sb-foot { padding:16px 20px;font-size:10px;color:#2d4a6a; }

/* Hide the group label ("nav" heading) */
section[data-testid="stSidebar"] .stRadio > label { display:none !important; }

/* Style each nav option label */
section[data-testid="stSidebar"] .stRadio label {
    display:flex !important;align-items:center;
    padding:9px 20px !important;color:#7a9ab8 !important;
    font-size:13px !important;font-weight:400 !important;
    border-left:3px solid transparent;border-radius:0 !important;margin:1px 0 !important;
}
section[data-testid="stSidebar"] .stRadio label:hover {
    background:rgba(255,255,255,0.05) !important;color:#aaccee !important;
}
section[data-testid="stSidebar"] .stRadio label:has(input:checked) {
    background:linear-gradient(90deg,rgba(56,189,248,0.12),transparent) !important;
    color:#38bdf8 !important;font-weight:600 !important;border-left-color:#38bdf8 !important;
}

/* Hide ONLY the radio circle element, keep text visible */
section[data-testid="stSidebar"] .stRadio [data-baseweb="radio"] { display:none !important; }
section[data-testid="stSidebar"] .stRadio [class*="radioMarkContainer"] { display:none !important; }

/* Sidebar nav buttons */
.nav-btn {
    display:block;width:100%;text-align:left;
    padding:9px 20px;font-size:13px;color:#7a9ab8;
    background:transparent;border:none;border-left:3px solid transparent;
    cursor:pointer;margin:1px 0;font-family:'Inter',sans-serif;
    text-decoration:none;
}
.nav-btn:hover { background:rgba(255,255,255,0.05);color:#aaccee; }
.nav-btn.active {
    background:linear-gradient(90deg,rgba(56,189,248,0.12),transparent);
    color:#38bdf8;font-weight:600;border-left-color:#38bdf8;
}

/* Dark labels for form inputs in main content */
.block-container [data-testid="stWidgetLabel"] p,
.block-container [data-testid="stWidgetLabel"] label,
.block-container .stSlider label,
.block-container .stSelectbox label,
.block-container .stNumberInput label {
    color:#1e293b !important;
}
/* Placeholder text */
.placeholder-text { color:#64748b !important; }

/* Sidebar nav buttons — override Streamlit default button style */
section[data-testid="stSidebar"] .stButton > button {
    background:transparent !important;
    color:#7a9ab8 !important;
    border:none !important;
    border-left:3px solid transparent !important;
    border-radius:0 !important;
    text-align:left !important;
    font-size:13px !important;
    font-weight:400 !important;
    padding:9px 20px !important;
    width:100% !important;
    box-shadow:none !important;
    justify-content:flex-start !important;
    display:flex !important;
    align-items:center !important;
}
section[data-testid="stSidebar"] .stButton > button:hover {
    background:rgba(255,255,255,0.05) !important;
    color:#aaccee !important;
}
section[data-testid="stSidebar"] .stButton > button:focus {
    background:linear-gradient(90deg,rgba(56,189,248,0.12),transparent) !important;
    color:#38bdf8 !important;
    border-left:3px solid #38bdf8 !important;
    font-weight:600 !important;
}

/* ── Layout ── */
.block-container { padding:24px 28px 40px !important;max-width:100% !important; }

/* ── Header ── */
.pg-crumb { font-size:10px;color:#94a3b8;letter-spacing:1.5px;text-transform:uppercase;margin-bottom:4px; }
.pg-title { font-size:26px;font-weight:700;color:#0f172a;margin:0 0 2px; }
.pg-sub   { font-size:12px;color:#64748b;margin-bottom:0; }
.badge    { display:inline-block;font-size:10px;font-weight:700;letter-spacing:1px;padding:3px 10px;border-radius:20px;margin-right:6px; }
.badge-live { background:#dcfce7;color:#15803d; }
.badge-ml   { background:#ede9fe;color:#6d28d9; }

/* ── Metric Card ── */
.mc { background:#fff;border-radius:10px;padding:18px 20px;box-shadow:0 1px 4px rgba(0,0,0,.07);border-top:3px solid;height:100%; }
.mc.g { border-top-color:#22c55e; }
.mc.b { border-top-color:#3b82f6; }
.mc.y { border-top-color:#f59e0b; }
.mc.p { border-top-color:#8b5cf6; }
.mc.r { border-top-color:#ef4444; }
.mc-lbl { font-size:9px;font-weight:700;letter-spacing:1.5px;text-transform:uppercase;color:#94a3b8;margin-bottom:8px; }
.mc-val { font-size:34px;font-weight:800;color:#0f172a;line-height:1;margin-bottom:4px; }
.mc-sub { font-size:11px;color:#64748b; }

/* ── Chart / Content Card ── */
.cc { background:#fff;border-radius:10px;padding:20px 22px;box-shadow:0 1px 4px rgba(0,0,0,.07);margin-bottom:0; }
.cc-title { font-size:10px;font-weight:700;letter-spacing:1.5px;text-transform:uppercase;color:#64748b;margin-bottom:14px; }

/* ── Result Card ── */
.res { border-radius:10px;padding:22px 24px;margin-top:16px; }
.res.h { background:#f0fdf4;border:1.5px solid #86efac; }
.res.m { background:#fefce8;border:1.5px solid #fde047; }
.res.l { background:#fff1f2;border:1.5px solid #fca5a5; }
.res-lbl { font-size:9px;font-weight:700;letter-spacing:2px;text-transform:uppercase;color:#6b7280;margin-bottom:6px; }
.res-val { font-size:28px;font-weight:800;color:#111827;margin-bottom:8px; }
.res-txt { font-size:13px;color:#374151;line-height:1.6; }

/* ── Stat Row ── */
.srow { display:flex;gap:12px;margin-top:16px;flex-wrap:wrap; }
.sbox { background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;padding:10px 16px;flex:1;min-width:90px; }
.sk { font-size:9px;color:#94a3b8;font-weight:600;letter-spacing:1px;text-transform:uppercase;margin-bottom:4px; }
.sv { font-size:15px;font-weight:700;color:#0f172a; }

/* ── Buttons ── */
.stButton > button {
    background:#1d4ed8 !important;color:#fff !important;
    border:none !important;border-radius:8px !important;
    font-weight:600 !important;font-size:13px !important;padding:10px 28px !important;
}
.stButton > button:hover { background:#1e40af !important; }

/* ── Hide chrome ── */
#MainMenu, footer { visibility:hidden !important; }
header[data-testid="stHeader"] { display:none !important; }
</style>
""", unsafe_allow_html=True)

# ── SESSION STATE ─────────────────────────────────────────────
if "page" not in st.session_state:
    st.session_state.page = "Predict Demand"

NAV_PAGES = ["Predict Demand", "Overview", "Model Performance", "Project Info"]

# ── SIDEBAR ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sb-logo">LIBRAIQ</div>', unsafe_allow_html=True)
    st.markdown('<div class="sb-tag">Library Intelligence Analytics</div>', unsafe_allow_html=True)
    st.markdown('<div class="sb-sec">Navigation</div>', unsafe_allow_html=True)
    for p in NAV_PAGES:
        active_cls = "active" if st.session_state.page == p else ""
        if st.sidebar.button(p, key=f"nav_{p}", use_container_width=True):
            st.session_state.page = p
            st.rerun()
    st.markdown('<hr class="sb-hr"/>', unsafe_allow_html=True)
    st.markdown('<div class="sb-foot">v1.0 &nbsp;|&nbsp; ML Enabled</div>', unsafe_allow_html=True)

page = st.session_state.page


# ══════════════════════════════════════════════════════════════
# PAGE 1 — PREDICT DEMAND
# ══════════════════════════════════════════════════════════════
if page == "Predict Demand":
    col_h, col_b = st.columns([3, 1])
    with col_h:
        st.markdown('<div class="pg-crumb">LIBRAIQ / PREDICT</div>', unsafe_allow_html=True)
        st.markdown('<div class="pg-title">Predict Book Demand</div>', unsafe_allow_html=True)
        st.markdown('<div class="pg-sub">Select a school and subject to generate a demand forecast</div>', unsafe_allow_html=True)
    with col_b:
        st.write("")
        st.markdown('<span class="badge badge-live">LIVE</span><span class="badge badge-ml">ML ENABLED</span>', unsafe_allow_html=True)

    st.write("")

    st.write("")

    col_form, col_out = st.columns([1, 1.3])

    with col_form:
        st.markdown('<div class="cc"><div class="cc-title">Input Parameters</div>', unsafe_allow_html=True)
        school   = st.selectbox("School", list(SCHOOLS.keys()))
        subject  = st.selectbox("Subject", SCHOOLS[school])
        semester = st.slider("Target Semester", 1, 8, 4)
        borrow   = st.number_input("Est. Past Borrow Count", 0, 500, 25)
        relevance = st.selectbox("Course Relevance", [0, 1], index=1,
                                 format_func=lambda x: "Low Relevance" if x == 0 else "High Relevance")
        st.write("")
        predict_btn = st.button("Generate Prediction", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_out:
        if predict_btn:
            code  = int(model.predict(np.array([[ENC[subject], semester, borrow, relevance]]))[0])
            label = LABELS[code]
            cls   = "h" if code == 2 else ("m" if code == 1 else "l")

            if code == 2:
                rec = f"Books on <b>{subject}</b> are in HIGH demand. Acquire additional copies immediately."
            elif code == 1:
                rec = f"Books on <b>{subject}</b> have MODERATE demand. Maintain current stock and monitor trends."
            else:
                rec = f"Books on <b>{subject}</b> have LOW demand. Minimal acquisition is needed this semester."

            school_short = school.split("(")[0].strip()
            rel_str = "High" if relevance == 1 else "Low"

            st.markdown(f"""
            <div class="cc">
                <div class="cc-title">Prediction Result</div>
                <div class="res {cls}">
                    <div class="res-lbl">Predicted Demand Level</div>
                    <div class="res-val">{label} Demand</div>
                    <div class="res-txt">{rec}</div>
                </div>
                <div class="srow">
                    <div class="sbox"><div class="sk">Subject</div><div class="sv">{subject}</div></div>
                    <div class="sbox"><div class="sk">School</div><div class="sv">{school_short}</div></div>
                    <div class="sbox"><div class="sk">Semester</div><div class="sv">{semester}</div></div>
                </div>
                <div class="srow">
                    <div class="sbox"><div class="sk">Borrow Count</div><div class="sv">{borrow}</div></div>
                    <div class="sbox"><div class="sk">Relevance</div><div class="sv">{rel_str}</div></div>
                    <div class="sbox"><div class="sk">Demand Code</div><div class="sv">{code}</div></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.write("")
            st.markdown('<div class="cc"><div class="cc-title">School-Wide Demand Comparison</div>', unsafe_allow_html=True)
            rows = []
            for s in SCHOOLS[school]:
                c = int(model.predict(np.array([[ENC[s], semester, borrow, relevance]]))[0])
                rows.append({"Subject": s, "Demand": LABELS[c], "Code": c})
            df_s = pd.DataFrame(rows).set_index("Subject")
            counts = df_s["Demand"].value_counts().reindex(["High", "Medium", "Low"], fill_value=0)
            st.bar_chart(counts, color="#3b82f6")
            st.dataframe(df_s[["Demand"]], use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="cc" style="min-height:320px;display:flex;align-items:center;justify-content:center;text-align:center;">
                <div>
                    <div style="font-size:48px;color:#e2e8f0;margin-bottom:12px;">&#9632;</div>
                    <div style="font-size:14px;font-weight:600;color:#475569;">Configure inputs on the left</div>
                    <div style="font-size:12px;color:#64748b;margin-top:4px;">and click Generate Prediction</div>
                </div>
            </div>
            """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# PAGE 2 — OVERVIEW
# ══════════════════════════════════════════════════════════════
elif page == "Overview":
    col_h, col_b = st.columns([3, 1])
    with col_h:
        st.markdown('<div class="pg-crumb">LIBRAIQ / OVERVIEW</div>', unsafe_allow_html=True)
        st.markdown('<div class="pg-title">System Overview</div>', unsafe_allow_html=True)
        st.markdown('<div class="pg-sub">Academic Batch 2025-26 &nbsp;|&nbsp; Live Dashboard</div>', unsafe_allow_html=True)
    with col_b:
        st.write("")
        st.markdown('<span class="badge badge-live">LIVE</span><span class="badge badge-ml">ML ENABLED</span>', unsafe_allow_html=True)

    st.write("")
    df_all = predict_all()
    high_c = int((df_all["Code"] == 2).sum())
    med_c  = int((df_all["Code"] == 1).sum())
    low_c  = int((df_all["Code"] == 0).sum())
    total  = len(df_all)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f'<div class="mc g"><div class="mc-lbl">Total Schools</div><div class="mc-val">{len(SCHOOLS)}</div><div class="mc-sub">{total} subjects tracked</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="mc b"><div class="mc-lbl">High Demand</div><div class="mc-val">{high_c}</div><div class="mc-sub">{high_c/total*100:.0f}% of subjects</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="mc y"><div class="mc-lbl">Medium Demand</div><div class="mc-val">{med_c}</div><div class="mc-sub">{med_c/total*100:.0f}% of subjects</div></div>', unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="mc r"><div class="mc-lbl">Low Demand</div><div class="mc-val">{low_c}</div><div class="mc-sub">{low_c/total*100:.0f}% of subjects</div></div>', unsafe_allow_html=True)

    st.write("")
    col_l, col_r = st.columns([1.3, 1])
    with col_l:
        st.markdown('<div class="cc"><div class="cc-title">Demand Distribution by School</div>', unsafe_allow_html=True)
        pivot = df_all.groupby(["School", "Demand"]).size().unstack(fill_value=0)
        for col in ["High", "Medium", "Low"]:
            if col not in pivot.columns:
                pivot[col] = 0
        pivot = pivot[["High", "Medium", "Low"]]
        st.bar_chart(pivot, color=["#22c55e", "#f59e0b", "#ef4444"])
        st.markdown('</div>', unsafe_allow_html=True)
    with col_r:
        st.markdown('<div class="cc"><div class="cc-title">Overall Demand Breakdown</div>', unsafe_allow_html=True)
        summary = df_all["Demand"].value_counts().reindex(["High", "Medium", "Low"], fill_value=0)
        st.bar_chart(summary, color="#3b82f6")
        st.markdown('</div>', unsafe_allow_html=True)

    st.write("")
    st.markdown('<div class="cc"><div class="cc-title">All Subjects — Demand Status</div>', unsafe_allow_html=True)
    st.dataframe(df_all[["School", "Subject", "Demand"]], use_container_width=True, hide_index=True)
    st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# PAGE 3 — MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════
elif page == "Model Performance":
    st.markdown('<div class="pg-crumb">LIBRAIQ / MODEL PERFORMANCE</div>', unsafe_allow_html=True)
    st.markdown('<div class="pg-title">Model Performance</div>', unsafe_allow_html=True)
    st.markdown('<div class="pg-sub">Comparative evaluation of all trained ML models</div>', unsafe_allow_html=True)
    st.write("")

    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH) as f:
            metrics = json.load(f)

        best_name = max(metrics, key=lambda k: metrics[k]["F1-score"])
        best = metrics[best_name]

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f'<div class="mc g"><div class="mc-lbl">Best Accuracy</div><div class="mc-val">{best["Accuracy"]*100:.1f}%</div><div class="mc-sub">{best_name}</div></div>', unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="mc b"><div class="mc-lbl">Best F1-Score</div><div class="mc-val">{best["F1-score"]:.3f}</div><div class="mc-sub">{best_name}</div></div>', unsafe_allow_html=True)
        with c3:
            st.markdown(f'<div class="mc y"><div class="mc-lbl">Best Precision</div><div class="mc-val">{best["Precision"]:.3f}</div><div class="mc-sub">{best_name}</div></div>', unsafe_allow_html=True)
        with c4:
            st.markdown(f'<div class="mc p"><div class="mc-lbl">Best Recall</div><div class="mc-val">{best["Recall"]:.3f}</div><div class="mc-sub">{best_name}</div></div>', unsafe_allow_html=True)

        st.write("")
        col_l, col_r = st.columns([1.4, 1])

        with col_l:
            st.markdown('<div class="cc"><div class="cc-title">All Models — Performance Metrics</div>', unsafe_allow_html=True)
            rows = [{"Model": n, "Accuracy": m["Accuracy"], "F1-Score": m["F1-score"],
                     "Precision": m["Precision"], "Recall": m["Recall"]}
                    for n, m in metrics.items()]
            df_m = pd.DataFrame(rows).set_index("Model")
            styled = df_m.style.highlight_max(axis=0, color='rgba(34,197,94,0.25)').format("{:.4f}")
            st.dataframe(styled, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with col_r:
            st.markdown('<div class="cc"><div class="cc-title">Confusion Matrix</div>', unsafe_allow_html=True)
            sel = st.selectbox("Select Model", list(metrics.keys()))
            cm = metrics[sel]["Confusion Matrix"]
            labels = ["Low", "Medium", "High"] if len(cm) == 3 else [f"Class {i}" for i in range(len(cm))]
            cm_df = pd.DataFrame(cm,
                                 index=[f"Actual {l}" for l in labels],
                                 columns=[f"Pred {l}" for l in labels])
            styled_cm = cm_df.style.background_gradient(cmap='Blues', axis=None).format("{:.0f}")
            st.dataframe(styled_cm, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.warning("Model metrics not found. Please train the model first.")


# ══════════════════════════════════════════════════════════════
# PAGE 4 — PROJECT INFO
# ══════════════════════════════════════════════════════════════
elif page == "Project Info":
    st.markdown('<div class="pg-crumb">LIBRAIQ / PROJECT</div>', unsafe_allow_html=True)
    st.markdown('<div class="pg-title">Project Information</div>', unsafe_allow_html=True)
    st.markdown('<div class="pg-sub">About this ML pipeline and future roadmap</div>', unsafe_allow_html=True)
    st.write("")

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("""
        <div class="cc">
            <div class="cc-title">About LibraIQ</div>
            <p style="font-size:13px;color:#374151;line-height:1.8;">
                <b>LibraIQ</b> is an end-to-end Machine Learning system built to predict demand
                levels for university library books based on academic and usage data.
            </p>
            <p style="font-size:13px;color:#374151;line-height:1.8;">
                The system classifies each subject's demand as <b>High</b>, <b>Medium</b>, or
                <b>Low</b>, enabling data-driven acquisition decisions.
            </p>
            <div class="cc-title" style="margin-top:16px;">ML Pipeline</div>
            <ul style="font-size:13px;color:#374151;line-height:2;padding-left:18px;">
                <li>Synthetic data generation</li>
                <li>Feature engineering and encoding</li>
                <li>Multi-model training and evaluation</li>
                <li>Best model selection and persistence</li>
                <li>Interactive Streamlit dashboard</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col_b:
        st.markdown("""
        <div class="cc">
            <div class="cc-title">Folder Structure</div>
            <div style="font-family:monospace;font-size:12px;color:#374151;line-height:2;
                        background:#f8fafc;padding:14px;border-radius:8px;margin-bottom:16px;">
                library-book-demand-prediction/<br>
                &nbsp;&nbsp;data/&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&#8212; Dataset<br>
                &nbsp;&nbsp;scripts/&nbsp;&nbsp;&#8212; Model training<br>
                &nbsp;&nbsp;models/&nbsp;&nbsp;&#8212; Saved models<br>
                &nbsp;&nbsp;dashboard/&#8212; Streamlit app
            </div>
            <div class="cc-title">Future Scope</div>
            <ul style="font-size:13px;color:#374151;line-height:2;padding-left:18px;">
                <li>Real library data integration</li>
                <li>Bulk CSV upload prediction</li>
                <li>Automatic periodic retraining</li>
                <li>Library management system integration</li>
                <li>Web service deployment</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

# ── FOOTER ────────────────────────────────────────────────────
st.divider()
st.caption("LibraIQ — Built as an academic Machine Learning project.")
