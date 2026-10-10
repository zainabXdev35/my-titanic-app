from pathlib import Path
import pickle
 
import pandas as pd
import streamlit as st
 
st.set_page_config(
    page_title="Titanic Survival Predictor",
    page_icon=":material/directions_boat:",
    layout="wide",
)
 
BASE_DIR = Path(__file__).resolve().parent
ARTIFACT_PATH = BASE_DIR / "titanic_artifacts.pkl"
 
PCLASS_OPTIONS = ["First", "Second", "Third"]
GENDER_OPTIONS = ["Female", "Male"]
SIBLING_OPTIONS = ["Zero", "One", "Two", "Three"]
EMBARKED_OPTIONS = ["Cherbourg", "Southampton", "Queenstown"]
FEATURE_COLUMNS = ["PClass", "Gender", "Sibling", "Embarked"]
 
# Accent color for each ticket class on the boarding pass
CLASS_ACCENT = {"First": "#1FB6C9", "Second": "#5B7FB5", "Third": "#7C8B9C"}
 
# ---------------------------------------------------------------------------
# Styling
# Palette: abyss #06182A, deep sea #0E3550, glacier #1FB6C9,
#          ice #F1F7FA, card white #FFFFFF, ink #0F2A3F,
#          survived #0F9D76, lost #D6454B
#
# Everything below sets its own colors, so the page looks the same whether
# Streamlit is in light or dark mode. No config file is needed.
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,600;12..96,800&family=Manrope:wght@400;500;600;700&display=swap');
 
:root {
    --abyss: #06182A;
    --sea: #0E3550;
    --glacier: #1FB6C9;
    --ice: #F1F7FA;
    --ink: #0F2A3F;
    --line: #D3E2EB;
    --chip: #E8F1F6;
    --muted: #4F667A;
    --ok: #0F9D76;
    --bad: #D6454B;
}
 
html, body, .stApp, [class*="css"] { font-family: 'Manrope', sans-serif; }
.stApp { background: var(--ice) !important; color: var(--ink); color-scheme: light; }
.block-container { max-width: 1120px; padding-top: 1.6rem; }
header[data-testid="stHeader"] { background: transparent; }
 
/* Sidebar (dark, light text) */
section[data-testid="stSidebar"] { background: var(--abyss) !important; }
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] li,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] label { color: #CFE0EA !important; }
section[data-testid="stSidebar"] strong { color: #FFFFFF !important; }
 
/* Hero */
.hero {
    position: relative;
    overflow: hidden;
    background:
        radial-gradient(circle at 88% 120%, rgba(31,182,201,.35) 0, transparent 45%),
        linear-gradient(125deg, var(--abyss) 0%, var(--sea) 100%);
    border-radius: 18px;
    padding: 2.3rem 2.4rem;
    margin-bottom: 1.5rem;
}
.hero::after {
    content: "";
    position: absolute;
    right: 2.2rem; bottom: 0;
    width: 190px; height: 120px;
    background: linear-gradient(160deg, #EAF8FB 0%, #8FD8E4 100%);
    clip-path: polygon(0 100%, 22% 48%, 38% 62%, 58% 8%, 78% 54%, 90% 40%, 100% 100%);
    opacity: .9;
}
.stApp div.hero h1 {
    font-family: 'Bricolage Grotesque', sans-serif;
    font-weight: 800;
    font-size: 2.4rem;
    line-height: 1.12;
    color: #FFFFFF !important;
    margin: 0 0 .6rem;
    padding: 0;
    max-width: 640px;
}
.stApp div.hero p { color: #A9C4D4 !important; margin: 0; max-width: 560px; font-size: 1.02rem; }
 
/* The two columns themselves become white cards */
div[data-testid="stColumn"],
div[data-testid="column"] {
    background: #FFFFFF;
    border: 1px solid var(--line);
    border-radius: 16px;
    padding: 1.3rem 1.4rem !important;
    box-sizing: border-box;
}
 
.panel-title {
    font-family: 'Bricolage Grotesque', sans-serif;
    font-weight: 800;
    font-size: 1.5rem;
    color: var(--abyss) !important;
    margin: 0 0 .9rem;
}
 
/* Text inside the main area */
[data-testid="stMain"] [data-testid="stMarkdownContainer"] p { color: var(--ink) !important; }
[data-testid="stMain"] [data-testid="stWidgetLabel"] p,
[data-testid="stMain"] [data-testid="stWidgetLabel"] label { color: var(--ink) !important; font-weight: 700; }
[data-testid="stMain"] [data-testid="stCaptionContainer"],
[data-testid="stMain"] [data-testid="stCaptionContainer"] p { color: var(--muted) !important; }
[data-testid="stMain"] [data-testid="stAlert"] p,
[data-testid="stMain"] [data-testid="stAlert"] div { color: var(--ink) !important; }
[data-testid="stMain"] [data-testid="stSpinner"] p,
[data-testid="stMain"] [data-testid="stSpinner"] div { color: var(--ink) !important; }
[data-testid="stMain"] code { color: var(--sea) !important; background: var(--chip) !important; }
 
/* Expander */
[data-testid="stMain"] [data-testid="stExpander"] details {
    background: var(--ice);
    border: 1px solid var(--line);
    border-radius: 12px;
}
[data-testid="stMain"] [data-testid="stExpander"] summary,
[data-testid="stMain"] [data-testid="stExpander"] summary p,
[data-testid="stMain"] [data-testid="stExpander"] summary span { color: var(--ink) !important; font-weight: 600; }
[data-testid="stMain"] [data-testid="stExpander"] summary svg { fill: var(--ink); color: var(--ink); }
 
/* Radio buttons restyled as selectable chips */
div[role="radiogroup"] { gap: .5rem; flex-wrap: wrap; }
div[role="radiogroup"] label {
    background: var(--chip);
    border: 1.5px solid var(--line);
    border-radius: 999px;
    padding: .4rem 1.05rem;
    margin: 0;
    cursor: pointer;
    transition: border-color .15s ease, transform .15s ease, background .15s ease;
}
div[role="radiogroup"] label > div:first-child { display: none; }
div[role="radiogroup"] label p,
div[role="radiogroup"] label span { color: var(--ink) !important; font-weight: 600; }
div[role="radiogroup"] label:hover { border-color: var(--glacier); transform: translateY(-1px); }
div[role="radiogroup"] label:has(input:checked) { background: var(--abyss); border-color: var(--abyss); }
div[role="radiogroup"] label:has(input:checked) p,
div[role="radiogroup"] label:has(input:checked) span { color: #FFFFFF !important; }
div[role="radiogroup"] label:has(input:focus-visible) { outline: 3px solid var(--glacier); outline-offset: 2px; }
 
/* Predict button */
div.stButton > button[kind="primary"],
div.stButton > button[data-testid="stBaseButton-primary"] {
    width: 100%;
    min-height: 3rem;
    border: 0;
    border-radius: 12px;
    background: linear-gradient(90deg, var(--sea), var(--glacier)) !important;
    color: #FFFFFF !important;
    font-weight: 700;
    font-size: 1.02rem;
    transition: transform .15s ease, box-shadow .15s ease;
}
div.stButton > button p { color: #FFFFFF !important; }
div.stButton > button[kind="primary"]:hover,
div.stButton > button[data-testid="stBaseButton-primary"]:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 18px rgba(14,53,80,.28);
}
 
/* Boarding pass (updates live as inputs change) */
.pass {
    display: flex;
    background: #FFFFFF;
    border: 1px solid var(--line);
    border-radius: 14px;
    overflow: hidden;
    margin-bottom: 1.4rem;
    box-shadow: 0 4px 14px rgba(14,53,80,.08);
}
.pass-main { flex: 1; padding: 1rem 1.2rem; }
.pass-title { font-family: 'Bricolage Grotesque', sans-serif; font-weight: 700; color: var(--abyss); font-size: 1.1rem; margin-bottom: .7rem; }
.pass-grid { display: grid; grid-template-columns: 1fr 1fr; gap: .7rem 1rem; }
.pass-grid .k { color: var(--muted); font-size: .8rem; }
.pass-grid .v { color: var(--abyss); font-weight: 700; font-size: 1rem; }
.pass-stub {
    width: 120px;
    border-left: 2px dashed var(--line);
    padding: 1rem .6rem;
    text-align: center;
    color: #FFFFFF;
    display: flex; flex-direction: column; justify-content: center;
    transition: background .25s ease;
}
.pass-stub .k { font-size: .75rem; opacity: .9; }
.pass-stub .v { font-family: 'Bricolage Grotesque', sans-serif; font-weight: 800; font-size: 1.15rem; }
 
/* Result */
@keyframes pop { from { opacity: 0; transform: translateY(10px) scale(.97); } to { opacity: 1; transform: none; } }
@keyframes fill { from { width: 0; } }
 
.verdict { border-radius: 14px; padding: 1.3rem 1.4rem; color: #FFFFFF; margin-bottom: 1rem; animation: pop .35s ease-out; }
.verdict.ok { background: linear-gradient(120deg, #0B7F5F, var(--ok)); }
.verdict.bad { background: linear-gradient(120deg, #B3363C, var(--bad)); }
.verdict .kicker { font-size: .88rem; opacity: .9; }
.verdict .headline { font-family: 'Bricolage Grotesque', sans-serif; font-weight: 800; font-size: 1.9rem; }
 
.meter-label { display: flex; justify-content: space-between; color: var(--ink); font-weight: 700; font-size: .92rem; margin-bottom: .35rem; }
.meter { height: 14px; background: #DDE8EF; border-radius: 99px; overflow: hidden; }
.meter > span { display: block; height: 100%; border-radius: 99px; animation: fill .9s ease-out; }
 
.hint { color: var(--muted); line-height: 1.6; }
 
/* Input summary table */
.sumtable { width: 100%; border-collapse: collapse; font-size: .95rem; }
.sumtable th { text-align: left; color: var(--muted); font-weight: 600; padding: .45rem .3rem; border-bottom: 1px solid var(--line); }
.sumtable td { color: var(--ink); padding: .5rem .3rem; border-bottom: 1px solid #E6EEF3; }
.sumtable td:last-child { font-weight: 700; }
.sumtable tr:last-child td { border-bottom: none; }
 
@media (prefers-reduced-motion: reduce) {
    .verdict, .meter > span { animation: none; }
}
</style>
""", unsafe_allow_html=True)
 
 
@st.cache_resource(show_spinner=False)
def load_artifacts():
    """Load the trusted model and fitted encoders saved from the notebook."""
    if not ARTIFACT_PATH.exists():
        return None
    # Only load pickle files you created and trust.
    with ARTIFACT_PATH.open("rb") as file:
        artifacts = pickle.load(file)
 
    required = {
        "model", "pclass_encoder", "gender_encoder",
        "sibling_encoder", "embarked_encoder"
    }
    missing = required.difference(artifacts.keys())
    if missing:
        raise ValueError("Artifact file missing: " + ", ".join(sorted(missing)))
    return artifacts
 
 
# ---------------------------------------------------------------------------
# Header and sidebar
# ---------------------------------------------------------------------------
st.markdown("""
<div class="hero">
<h1>Would this passenger have survived the Titanic?</h1>
<p>Enter four passenger details to get a prediction from the trained SVC model.</p>
</div>
""", unsafe_allow_html=True)
 
with st.sidebar:
    st.header("About")
    st.write("This educational app uses a trained machine-learning model to predict Titanic passenger survival.")
    st.markdown("**Features used**")
    st.markdown("- Passenger class\n- Gender\n- Siblings/spouse category\n- Port of embarkation")
    st.caption("Predictions are estimates based on historical data, not certainties.")
 
# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
input_col, result_col = st.columns(2, gap="large")
 
with input_col:
    st.markdown('<div class="panel-title">Passenger details</div>', unsafe_allow_html=True)
    pclass_input = st.radio("Passenger class", PCLASS_OPTIONS, horizontal=True, key="pclass")
    gender_input = st.radio("Gender", GENDER_OPTIONS, index=1, horizontal=True, key="gender")
    sibling_input = st.radio("Siblings/spouse aboard", SIBLING_OPTIONS, horizontal=True, key="sibling")
    embarked_input = st.radio("Port of embarkation", EMBARKED_OPTIONS, index=1, horizontal=True, key="embarked")
    st.write("")
    submitted = st.button("Predict survival", type="primary")
 
with result_col:
    # Live boarding pass: refreshes every time a choice on the left changes
    accent = CLASS_ACCENT[pclass_input]
    st.markdown(f"""
<div class="pass">
<div class="pass-main">
<div class="pass-title">Passenger boarding pass</div>
<div class="pass-grid">
<div><div class="k">Class</div><div class="v">{pclass_input}</div></div>
<div><div class="k">Gender</div><div class="v">{gender_input}</div></div>
<div><div class="k">Siblings/spouse</div><div class="v">{sibling_input}</div></div>
<div><div class="k">Port</div><div class="v">{embarked_input}</div></div>
</div>
</div>
<div class="pass-stub" style="background:{accent};">
<div class="k">Class</div>
<div class="v">{pclass_input}</div>
</div>
</div>
""", unsafe_allow_html=True)
 
    st.markdown('<div class="panel-title">Prediction</div>', unsafe_allow_html=True)
    if not ARTIFACT_PATH.exists():
        st.warning("The model artifact file is missing.")
        st.write("Upload `titanic_artifacts.pkl` to the same GitHub folder as `app.py`.")
    elif not submitted:
        st.markdown(
            '<div class="hint">Adjust the details and select <b>Predict survival</b>. '
            'The boarding pass above updates as you choose.</div>',
            unsafe_allow_html=True,
        )
 
    if submitted:
        try:
            artifacts = load_artifacts()
            if artifacts is None:
                st.error("Could not find titanic_artifacts.pkl beside app.py.")
            else:
                with st.spinner("Checking the passenger manifest..."):
                    user_input = pd.DataFrame({
                        "PClass": [pclass_input],
                        "Gender": [gender_input],
                        "Sibling": [sibling_input],
                        "Embarked": [embarked_input],
                    }, columns=FEATURE_COLUMNS)
 
                    encoded = user_input.copy()
                    encoded["PClass"] = artifacts["pclass_encoder"].transform(user_input["PClass"])
                    encoded["Gender"] = artifacts["gender_encoder"].transform(user_input["Gender"])
                    encoded["Sibling"] = artifacts["sibling_encoder"].transform(user_input["Sibling"])
                    encoded["Embarked"] = artifacts["embarked_encoder"].transform(user_input["Embarked"])
 
                    model = artifacts["model"]
                    prediction = int(model.predict(encoded)[0])
 
                survived = prediction == 1
                css = "ok" if survived else "bad"
                headline = "Survived" if survived else "Did not survive"
                st.markdown(f"""
<div class="verdict {css}">
<div class="kicker">Predicted outcome</div>
<div class="headline">{headline}</div>
</div>
""", unsafe_allow_html=True)
 
                if hasattr(model, "predict_proba"):
                    try:
                        probs = model.predict_proba(encoded)[0]
                        classes = list(model.classes_)
                        if 1 in classes:
                            p = float(probs[classes.index(1)])
                            pct = round(p * 100)
                            bar = "var(--ok)" if survived else "var(--bad)"
                            st.markdown(f"""
<div class="meter-label"><span>Model-estimated survival probability</span><span>{p:.1%}</span></div>
<div class="meter"><span style="width:{pct}%; background:{bar};"></span></div>
<div style="height:.8rem"></div>
""", unsafe_allow_html=True)
                    except (ValueError, AttributeError, IndexError):
                        pass
 
                with st.expander("Input summary"):
                    rows = [
                        ("Passenger class", pclass_input),
                        ("Gender", gender_input),
                        ("Siblings/spouse", sibling_input),
                        ("Port", embarked_input),
                    ]
                    body = "".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in rows)
                    st.markdown(
                        f'<table class="sumtable"><tr><th>Feature</th><th>Selected value</th></tr>{body}</table>',
                        unsafe_allow_html=True,
                    )
        except ValueError as exc:
            st.error("An input category was not recognized by the saved encoder. Make sure the options match the notebook.")
            st.caption(f"Technical detail: {exc}")
        except Exception as exc:
            st.error("The prediction could not be completed.")
            st.caption(f"Technical detail: {type(exc).__name__}: {exc}")
 
st.caption("For educational use only. Machine-learning predictions can be wrong.")
 
