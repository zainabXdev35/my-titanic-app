
import streamlit as st

# Page configuration
st.set_page_config(
    page_title="Titanic Survival Predictor",
    page_icon="🚢",
    layout="wide"
)

# Custom styling
st.markdown("""
<style>
.stApp {
    background-color: #f3f6fb;
}
.main-title {
    font-size: 38px;
    font-weight: 800;
    color: #16324f;
}
.subtitle {
    color: #526579;
    font-size: 17px;
}
div.stButton > button {
    width: 100%;
    border-radius: 10px;
    height: 48px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# Header
st.markdown(
    '<div class="main-title">🚢 Titanic Survival Predictor</div>',
    unsafe_allow_html=True
)
st.markdown(
    '<div class="subtitle">Explore how passenger details relate '
    'to survival on the Titanic.</div>',
    unsafe_allow_html=True
)

st.divider()

# Sidebar
with st.sidebar:
    st.header("ℹ️ About this project")
    st.write(
        "This application demonstrates a machine learning "
        "workflow for Titanic passenger survival prediction."
    )
    st.info(
        "Enter passenger details, review the summary, "
        "and click the prediction button."
    )
    st.caption("Machine Learning Assignment | AIC354")

# Input sections
st.subheader("👤 Passenger Information")

left, right = st.columns(2)

with left:
    name = st.text_input(
        "Passenger name (optional)",
        placeholder="e.g. Alex Smith"
    )

    age = st.slider(
        "Age",
        min_value=0,
        max_value=100,
        value=25
    )

    sex = st.selectbox(
        "Sex",
        ["female", "male"]
    )

with right:
    pclass = st.selectbox(
        "Ticket class",
        [1, 2, 3],
        format_func=lambda x: {
            1: "1st Class",
            2: "2nd Class",
            3: "3rd Class"
        }[x]
    )

    fare = st.number_input(
        "Ticket fare",
        min_value=0.0,
        max_value=1000.0,
        value=32.0,
        step=1.0
    )

    siblings = st.number_input(
        "Siblings / spouse aboard",
        min_value=0,
        max_value=10,
        value=0
    )

    parents = st.number_input(
        "Parents / children aboard",
        min_value=0,
        max_value=10,
        value=0
    )

st.divider()

# Passenger summary
st.subheader("📋 Passenger Summary")

col1, col2, col3 = st.columns(3)

col1.metric("Age", f"{age} years")
col2.metric("Ticket class", f"{pclass}")
col3.metric("Family members aboard", siblings + parents)

st.write(
    f"**Passenger:** {name.strip() if name.strip() else 'Not provided'}"
)

# Prediction area
st.divider()
st.subheader("🔮 Survival Prediction")

st.write(
    "When the trained model is connected, this button "
    "will use the passenger details to generate a prediction."
)

if st.button("Predict Survival", type="primary"):
    st.warning(
        "Demo mode: the trained machine learning model has not "
        "been connected yet. No survival prediction was made."
    )

# Footer
st.divider()
st.caption(
    "Educational demonstration only. Predictions are not "
    "historical certainty or a guarantee of survival."
)
