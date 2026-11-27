
import streamlit as st
import pandas as pd
import joblib

# ------------------------------------------------
# PAGE CONFIGURATION
# ------------------------------------------------

st.set_page_config(
    page_title="UHPFRC Strength Predictor",
    page_icon="🏗️",
    layout="wide"
)

st.title("UHPFRC Strength Prediction")
st.subheader(
    "Machine Learning-Based Compressive and Flexural Strength Prediction"
)

st.write(
    "Enter the mix-design parameters to predict the "
    "compressive and flexural strength of UHPFRC."
)

# ------------------------------------------------
# LOAD TRAINED MODELS
# ------------------------------------------------

@st.cache_resource
def load_models():
    return joblib.load("UHPFRC_Chained_MTL_Models.pkl")

try:
    models = load_models()
    cs_model = models["CS_model"]
    fs_model = models["FS_model"]
    feature_cols = models["feature_cols"]
    fs_feature_cols = models["FS_feature_cols"]

except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# ------------------------------------------------
# INPUT PARAMETERS
# ------------------------------------------------

st.header("1. Enter Mix Design Parameters")

st.info(
    "Use the same units and conventions as the original "
    "UHPFRC training dataset."
)

input_groups = [
    ("C", "Cement"),
    ("SF", "Silica Fume"),
    ("QP", "Quartz Powder"),
    ("FA", "Fly Ash"),
    ("SL", "Slag"),
    ("MK", "Metakaolin"),
    ("S", "Sand"),
    ("QS", "Quartz Sand"),
    ("W", "Water"),
    ("SP", "Superplasticizer"),
    ("Age", "Curing Age (days)"),
    ("L", "Fiber Length"),
    ("D", "Fiber Diameter"),
    ("BV", "Basalt Fiber Volume"),
    ("PPV", "Polypropylene Fiber Volume"),
    ("GV", "Glass Fiber Volume"),
    ("SSV", "Steel Fiber Volume")
]

input_values = {}

col1, col2, col3 = st.columns(3)

for i, (feature, label) in enumerate(input_groups):

    with [col1, col2, col3][i % 3]:

        input_values[feature] = st.number_input(
            label,
            min_value=0.0,
            value=0.0,
            step=0.1,
            key=feature
        )

# ------------------------------------------------
# PREDICTION
# ------------------------------------------------

st.divider()
st.header("2. Predict Strength")

if st.button("Predict Strength", type="primary"):

    input_df = pd.DataFrame(
        [[input_values[col] for col in feature_cols]],
        columns=feature_cols
    )

    # Stage 1: Compressive strength
    predicted_cs = cs_model.predict(input_df)[0]

    # Stage 2: Flexural strength
    fs_input = input_df.copy()
    fs_input["Predicted_CS"] = predicted_cs

    fs_input = fs_input[fs_feature_cols]

    predicted_fs = fs_model.predict(fs_input)[0]

    # Display results
    st.success("Prediction completed!")

    result1, result2 = st.columns(2)

    with result1:
        st.metric(
            "Predicted Compressive Strength",
            f"{predicted_cs:.2f} MPa"
        )

    with result2:
        st.metric(
            "Predicted Flexural Strength",
            f"{predicted_fs:.2f} MPa"
        )

    st.subheader("Input Summary")

    summary = input_df.copy()
    summary["Predicted CS (MPa)"] = predicted_cs
    summary["Predicted FS (MPa)"] = predicted_fs

    st.dataframe(summary, use_container_width=True)

# ------------------------------------------------
# FOOTER
# ------------------------------------------------

st.divider()

st.caption(
    "UHPFRC Strength Prediction Tool | "
    "Powered by Gradient Boosting and Chained Learning"
)

st.caption(
    "Research prototype. Predictions are estimates "
    "and should be validated experimentally."
)
