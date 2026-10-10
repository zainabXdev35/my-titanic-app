from pathlib import Path
 
import pandas as pd
import streamlit as st
 
# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_PATH = Path(__file__).parent / "model.pkl"
 
# Match these to how the model was trained. If your model is a Pipeline that
# handles raw text itself, set ENCODE_INPUTS = False.
ENCODE_INPUTS = True
SEX_CODES = {"female": 0, "male": 1}
PORT_CODES = {"Southampton": 0, "Cherbourg": 1, "Queenstown": 2}
 
# Column order must match the training data.
FEATURE_ORDER = ["Pclass", "Sex", "SibSp", "Embarked"]
 
CLASS_LABELS = {1: "First class", 2: "Second class", 3: "Third class"}
 
st.set_page_config(
    page_title="Titanic Survival Predictor",
    page_icon=":material/directions_boat:",
    layout="wide",
)
 
# ---------------------------------------------------------------------------
# Styling
#   Night sea (#0B1D2E), steel (#1F3A52), fog (#E9EEF2),
#   funnel buff (#D69A2D), survived green (#1F7A63), lost red (#A63A32)
# ---------------------------------------------------------------------------
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,600;9..144,800&family=Public+Sans:wght@400;500;600&display=swap');
 
:root {
    --night: #0B1D2E;
    --steel: #1F3A52;
    --fog: #E9EEF2;
    --buff: #D69A2D;
    --ok: #1F7A63;
    --bad: #A63A32;
    --muted: #5B6F82;
}
 
html, body, [class*="css"], .stApp {
    font-family: 'Public Sans', sans-serif;
}
.stApp { background: var(--fog); }
.block-container { padding-top: 1.5rem; max-width: 1150px; }
header[data-testid="stHeader"] { background: transparent; }
 
/* Hero */
.hero {
    background: linear-gradient(135deg, var(--night) 0%, var(--steel) 100%);
    border-bottom: 6px solid var(--buff);
    border-radius: 14px;
    padding: 2.4rem 2.6rem;
    margin-bottom: 1.6rem;
}
.hero h1 {
    font-family: 'Fraunces', serif;
    font-weight: 800;
    font-size: 2.7rem;
    line-height: 1.1;
    color: #FFFFFF;
    margin: 0 0 .6rem 0;
}
.hero p {
    color: #B9C8D6;
    font-size: 1.05rem;
    max-width: 640px;
    margin: 0;
}
 
/* Section headings */
.section-title {
    font-family: 'Fraunces', serif;
    font-weight: 600;
    font-size: 1.35rem;
    color: var(--night);
    margin: 0 0 .2rem 0;
}
.section-note {
    color: var(--muted);
    font-size: .93rem;
    margin-bottom: 1rem;
}
 
/* Bordered containers become white panels */
div[data-testid="stVerticalBlockBorderWrapper"] {
    background: #FFFFFF;
    border: 1px solid #D3DCE4;
    border-radius: 14px;
    padding: .6rem .8rem;
}
 
/* Form widgets */
label, .stRadio label p, .stSlider label p {
    color: var(--night) !important;
    font-weight: 600 !important;
}
div[data-baseweb="radio"] > div:first-child { border-color: var(--steel); }
 
/* Submit button */
div.stFormSubmitButton > button {
    width: 100%;
    height: 52px;
    border: none;
    border-radius: 10px;
    background: var(--night);
    color: #FFFFFF;
    font-weight: 600;
    font-size: 1.02rem;
    transition: background .15s ease;
}
div.stFormSubmitButton > button:hover {
    background: var(--buff);
    color: var(--night);
}
div.stFormSubmitButton > button:focus-visible {
    outline: 3px solid var(--buff);
    outline-offset: 2px;
}
 
/* Result */
.verdict {
    border-radius: 12px;
    padding: 1.4rem 1.5rem;
    color: #FFFFFF;
    margin-bottom: 1rem;
}
.verdict.ok  { background: var(--ok); }
.verdict.bad { background: var(--bad); }
.verdict .kicker { font-size: .9rem; opacity: .85; margin-bottom: .2rem; }
.verdict .headline {
    font-family: 'Fraunces', serif;
    font-weight: 800;
    font-size: 2rem;
    line-height: 1.15;
}
 
.meter-label {
    display: flex; justify-content: space-between;
    font-size: .9rem; color: var(--night); font-weight: 600;
    margin-bottom: .35rem;
}
.meter {
    height: 14px; background: #DCE4EB; border-radius: 99px; overflow: hidden;
}
.meter > span { display: block; height: 100%; border-radius: 99px; }
 
.summary-row {
    display: flex; justify-content: space-between;
    padding: .55rem 0; border-bottom: 1px solid #E3E9EE;
    font-size: .95rem;
}
.summary-row:last-child { border-bottom: none; }
.summary-row .k { color: var(--muted); }
.summary-row .v { color: var(--night); font-weight: 600; }
 
.placeholder {
    text-align: center; color: var(--muted);
    padding: 2.6rem 1rem; line-height: 1.6;
}
.placeholder strong { color: var(--night); }
 
.footer-note {
    color: var(--muted); font-size: .85rem; text-align: center;
    margin-top: 1.6rem;
}
</style>
""",
    unsafe_allow_html=True,
)
 
 
# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_model():
    """Load the trained model once. Returns None if the file is missing."""
    if not MODEL_PATH.exists():
        return None
    import joblib
 
    return joblib.load(MODEL_PATH)
 
 
def build_features(pclass: int, sex: str, sibsp: int, embarked: str) -> pd.DataFrame:
    """Create a one-row DataFrame in the same shape the model was trained on."""
    row = {
        "Pclass": pclass,
        "Sex": SEX_CODES[sex] if ENCODE_INPUTS else sex,
        "SibSp": sibsp,
        "Embarked": PORT_CODES[embarked] if ENCODE_INPUTS else embarked,
    }
    return pd.DataFrame([row])[FEATURE_ORDER]
 
 
def demo_estimate(pclass: int, sex: str, sibsp: int, embarked: str) -> float:
    """
    Rough illustrative score used only when no trained model file is present.
    It is a hand-written rule of thumb, not a trained model.
    """
    score = 0.35
    score += 0.38 if sex == "female" else -0.18
    score += {1: 0.17, 2: 0.04, 3: -0.10}[pclass]
    score += {0: 0.0, 1: 0.04, 2: 0.0, 3: -0.05}.get(min(sibsp, 3), -0.10)
    score += 0.05 if embarked == "Cherbourg" else 0.0
    return max(0.03, min(0.97, score))
 
 
def predict(pclass: int, sex: str, sibsp: int, embarked: str) -> dict:
    """Return the predicted outcome, its probability and the source used."""
    model = load_model()
 
    if model is None:
        prob = demo_estimate(pclass, sex, sibsp, embarked)
        return {"survived": prob >= 0.5, "prob": prob, "source": "demo"}
 
    X = build_features(pclass, sex, sibsp, embarked)
 
    if hasattr(model, "predict_proba"):
        classes = list(model.classes_)
        prob = float(model.predict_proba(X)[0][classes.index(1)])
        return {"survived": prob >= 0.5, "prob": prob, "source": "model"}
 
    label = int(model.predict(X)[0])
    return {"survived": label == 1, "prob": None, "source": "model"}
 
 
# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("About")
    st.write(
        "A machine learning model trained on the Titanic passenger "
        "records estimates whether a passenger survived the sinking "
        "in April 1912."
    )
 
    st.subheader("What the model uses")
    st.markdown(
        "- **Ticket class** (first, second or third)\n"
        "- **Sex**\n"
        "- **Siblings or spouse aboard**\n"
        "- **Port of embarkation**"
    )
 
    st.subheader("What it predicts")
    st.write("Whether the passenger survived or did not survive.")
 
    st.divider()
    if load_model() is None:
        st.warning(
            "No trained model file was found. Results shown are an "
            "illustrative estimate. Add `model.pkl` next to `app.py` "
            "to use the real model."
        )
    else:
        st.success("Trained model loaded.")
 
# ---------------------------------------------------------------------------
# Hero
# ---------------------------------------------------------------------------
st.markdown(
    """
<div class="hero">
    <h1>Would this passenger have survived the Titanic?</h1>
    <p>Choose a ticket class, sex, family aboard and the port of boarding.
    The model estimates the outcome from patterns in the historical passenger records.</p>
</div>
""",
    unsafe_allow_html=True,
)
 
# ---------------------------------------------------------------------------
# Main layout
# ---------------------------------------------------------------------------
form_col, result_col = st.columns([1.05, 1], gap="large")
 
with form_col:
    with st.container(border=True):
        st.markdown('<div class="section-title">Passenger details</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-note">These four details match what the model was trained on.</div>',
            unsafe_allow_html=True,
        )
 
        with st.form("passenger_form", border=False):
            pclass = st.radio(
                "Ticket class",
                options=[1, 2, 3],
                format_func=lambda c: CLASS_LABELS[c],
                horizontal=True,
                index=2,
            )
 
            sex = st.radio(
                "Sex",
                options=["female", "male"],
                format_func=str.capitalize,
                horizontal=True,
            )
 
            sibsp = st.slider(
                "Siblings or spouse aboard",
                min_value=0,
                max_value=8,
                value=0,
                help="Number of siblings plus spouse travelling with the passenger.",
            )
 
            embarked = st.radio(
                "Port of embarkation",
                options=list(PORT_CODES.keys()),
                horizontal=True,
                help="Where the passenger boarded the ship.",
            )
 
            submitted = st.form_submit_button("Predict survival")
 
if submitted:
    st.session_state["result"] = {
        "inputs": {
            "Ticket class": CLASS_LABELS[pclass],
            "Sex": sex.capitalize(),
            "Siblings or spouse": sibsp,
            "Port of embarkation": embarked,
        },
        **predict(pclass, sex, sibsp, embarked),
    }
 
with result_col:
    with st.container(border=True):
        st.markdown('<div class="section-title">Prediction</div>', unsafe_allow_html=True)
 
        result = st.session_state.get("result")
 
        if result is None:
            st.markdown(
                """
<div class="placeholder">
    <strong>No prediction yet.</strong><br>
    Fill in the passenger details and select <em>Predict survival</em>.
</div>
""",
                unsafe_allow_html=True,
            )
        else:
            survived = result["survived"]
            css = "ok" if survived else "bad"
            headline = "Survived" if survived else "Did not survive"
 
            st.markdown(
                f"""
<div class="verdict {css}">
    <div class="kicker">Predicted outcome</div>
    <div class="headline">{headline}</div>
</div>
""",
                unsafe_allow_html=True,
            )
 
            if result["prob"] is not None:
                pct = round(result["prob"] * 100)
                bar_color = "var(--ok)" if survived else "var(--bad)"
                st.markdown(
                    f"""
<div class="meter-label"><span>Probability of survival</span><span>{pct}%</span></div>
<div class="meter"><span style="width:{pct}%; background:{bar_color};"></span></div>
<div style="height:1.1rem"></div>
""",
                    unsafe_allow_html=True,
                )
 
            rows = "".join(
                f'<div class="summary-row"><span class="k">{k}</span>'
                f'<span class="v">{v}</span></div>'
                for k, v in result["inputs"].items()
            )
            st.markdown(rows, unsafe_allow_html=True)
 
            if result["source"] == "demo":
                st.caption(
                    "Illustrative estimate. The trained model file is not connected yet."
                )
 
# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown(
    '<div class="footer-note">Predictions come from historical patterns and describe '
    "likelihood, not certainty. For educational use only.</div>",
    unsafe_allow_html=True,
)
 
