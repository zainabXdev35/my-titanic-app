from pathlib import Path
import pickle

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Titanic Survival Predictor", page_icon="🚢", layout="wide")

BASE_DIR = Path(__file__).resolve().parent
ARTIFACT_PATH = BASE_DIR / "titanic_artifacts.pkl"

PCLASS_OPTIONS = ["First", "Second", "Third"]
GENDER_OPTIONS = ["Female", "Male"]
SIBLING_OPTIONS = ["Zero", "One", "Two", "Three"]
EMBARKED_OPTIONS = ["Cherbourg", "Southampton", "Queenstown"]
FEATURE_COLUMNS = ["PClass", "Gender", "Sibling", "Embarked"]

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Playfair+Display:wght@600;700&display=swap');
.stApp { background: #edf2f5; font-family: 'DM Sans', sans-serif; }
.block-container { max-width: 1120px; padding-top: 1.8rem; }
.hero { background: linear-gradient(130deg,#102638,#244a60); border-bottom: 5px solid #d5a348; border-radius: 16px; padding: 2rem; margin-bottom: 1.4rem; }
.hero h1 { color: white; font-family: 'Playfair Display',serif; font-size: 2.3rem; margin: 0 0 .5rem; }
.hero p { color: #d8e3eb; margin: 0; }
div[data-testid="stVerticalBlockBorderWrapper"] { background: white; border: 1px solid #d6e0e7; border-radius: 14px; }
div.stFormSubmitButton > button { background: #102638; color: white; border: 0; border-radius: 9px; min-height: 2.8rem; font-weight: 700; width: 100%; }
div.stFormSubmitButton > button:hover { background: #d5a348; color: #102638; }
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


st.markdown("""
<div class="hero">
  <h1>Would this passenger have survived the Titanic?</h1>
  <p>Enter four passenger details to get a prediction from the trained SVC model.</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("About this project")
    st.write("This educational app uses a trained machine-learning model to predict Titanic passenger survival.")
    st.markdown("**Features used**")
    st.markdown("- Passenger class\n- Gender\n- Siblings/spouse category\n- Port of embarkation")
    st.caption("Predictions are estimates based on historical data, not certainties.")

input_col, result_col = st.columns(2, gap="large")

with input_col:
    with st.container(border=True):
        st.subheader("Passenger details")
        with st.form("passenger_details"):
            pclass_input = st.selectbox("Passenger class", PCLASS_OPTIONS)
            gender_input = st.selectbox("Gender", GENDER_OPTIONS, index=1)
            sibling_input = st.selectbox("Siblings/spouse aboard", SIBLING_OPTIONS)
            embarked_input = st.selectbox("Port of embarkation", EMBARKED_OPTIONS, index=1)
            submitted = st.form_submit_button("Predict survival")

with result_col:
    with st.container(border=True):
        st.subheader("Prediction")
        if not ARTIFACT_PATH.exists():
            st.warning("The model artifact file is missing.")
            st.write("Upload `titanic_artifacts.pkl` to the same GitHub folder as `app.py`.")
        else:
            st.write("Choose the passenger details and click **Predict survival**.")

        if submitted:
            try:
                artifacts = load_artifacts()
                if artifacts is None:
                    st.error("Could not find titanic_artifacts.pkl beside app.py.")
                else:
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

                    if prediction == 1:
                        st.success("Prediction: SURVIVED")
                    else:
                        st.error("Prediction: NOT SURVIVED")

                    st.markdown("**Input summary**")
                    st.dataframe(pd.DataFrame({
                        "Feature": ["Passenger class", "Gender", "Siblings/spouse", "Port"],
                        "Selected value": [pclass_input, gender_input, sibling_input, embarked_input],
                    }), hide_index=True, use_container_width=True)

                    if hasattr(model, "predict_proba"):
                        try:
                            probs = model.predict_proba(encoded)[0]
                            classes = list(model.classes_)
                            if 1 in classes:
                                st.metric("Model-estimated survival probability", f"{float(probs[classes.index(1)]):.1%}")
                        except (ValueError, AttributeError, IndexError):
                            pass
            except ValueError as exc:
                st.error("An input category was not recognized by the saved encoder. Make sure the options match the notebook.")
                st.caption(f"Technical detail: {exc}")
            except Exception as exc:
                st.error("The prediction could not be completed.")
                st.caption(f"Technical detail: {type(exc).__name__}: {exc}")

st.caption("For educational use only. Machine-learning predictions can be wrong.")
