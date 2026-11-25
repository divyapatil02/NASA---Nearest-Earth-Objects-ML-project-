
"""
NASA Near-Earth Object (NEO) Hazard Predictor
Streamlit Front End

Pipeline:
    User Inputs
        -> Saved preprocessing pipeline
        -> RandomForestClassifier
        -> 0 / 1
        -> Hazardous / Non-Hazardous

Run:
    pip install -r requirements.txt
    streamlit run app.py

Keep model_prod.pkl in the same folder as app.py.
"""

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

MODEL_PATH = Path(__file__).parent / "model_prod.pkl"

FEATURES = [
    "absolute_magnitude",
    "est_diameter_min",
    "est_diameter_max",
    "relative_velocity",
    "miss_distance"
]

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


# --------------------------------------------------
# PREPROCESS INPUT
# --------------------------------------------------

def preprocess(raw, model):

    X = pd.DataFrame([raw])

    # Maintain the exact feature order used during training
    if hasattr(model, "feature_names_in_"):
        X = X[list(model.feature_names_in_)]

    return X


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="NASA NEO Hazard Predictor",
    page_icon="🚀",
    layout="centered"
)

st.title("🚀 NASA NEO Hazard Predictor")

st.write(
    """
    Predict whether a Near-Earth Object (asteroid)
    is potentially hazardous using Machine Learning.
    
    Enter the asteroid's physical and orbital details
    and click **Predict**.
    """
)

st.divider()

# --------------------------------------------------
# CHECK MODEL
# --------------------------------------------------

if not MODEL_PATH.exists():

    st.error(
        "model_prod.pkl not found. "
        "Place the trained model in the same folder as app.py."
    )

    st.stop()

try:
    model = load_model()

except Exception as e:

    st.error(f"Unable to load model: {e}")
    st.stop()


# --------------------------------------------------
# INPUT FORM
# --------------------------------------------------

st.subheader("Asteroid Details")

with st.form("neo_prediction_form"):

    left, right = st.columns(2)

    with left:

        absolute_magnitude = st.number_input(
            "Absolute Magnitude",
            min_value=-10.0,
            max_value=40.0,
            value=22.0,
            step=0.1
        )

        est_diameter_min = st.number_input(
            "Minimum Estimated Diameter (km)",
            min_value=0.0,
            max_value=100.0,
            value=0.05,
            step=0.01,
            format="%.4f"
        )

        est_diameter_max = st.number_input(
            "Maximum Estimated Diameter (km)",
            min_value=0.0,
            max_value=100.0,
            value=0.12,
            step=0.01,
            format="%.4f"
        )

    with right:

        relative_velocity = st.number_input(
            "Relative Velocity (km/h)",
            min_value=0.0,
            max_value=2000000.0,
            value=45000.0,
            step=100.0
        )

        miss_distance = st.number_input(
            "Miss Distance (km)",
            min_value=0.0,
            max_value=100000000.0,
            value=5000000.0,
            step=1000.0
        )

    submitted = st.form_submit_button(
        "🔍 Predict Hazard Status",
        type="primary",
        use_container_width=True
    )


# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

if submitted:

    if est_diameter_max < est_diameter_min:

        st.error(
            "Maximum diameter must be greater than "
            "or equal to minimum diameter."
        )

    else:

        raw = {
            "absolute_magnitude": absolute_magnitude,
            "est_diameter_min": est_diameter_min,
            "est_diameter_max": est_diameter_max,
            "relative_velocity": relative_velocity,
            "miss_distance": miss_distance
        }

        try:

            X = preprocess(raw, model)

            prediction = int(model.predict(X)[0])

            # Probability of predicted class
            if hasattr(model, "predict_proba"):

                probabilities = model.predict_proba(X)[0]

                class_index = list(model.classes_).index(
                    prediction
                )

                confidence = probabilities[class_index]

                hazardous_probability = (
                    probabilities[list(model.classes_).index(1)]
                    if 1 in model.classes_
                    else 0.0
                )

            else:

                confidence = None
                hazardous_probability = None

            # --------------------------------------
            # DISPLAY RESULTS
            # --------------------------------------

            st.divider()

            st.subheader("Prediction Result")

            if prediction == 1:

                st.error(
                    "⚠️ Potentially Hazardous Asteroid"
                )

                label = "Hazardous"

            else:

                st.success(
                    "✅ Non-Hazardous Asteroid"
                )

                label = "Non-Hazardous"

            col1, col2 = st.columns(2)

            col1.metric(
                "Model Output",
                f"{prediction} → {label}"
            )

            if confidence is not None:

                col2.metric(
                    "Prediction Confidence",
                    f"{confidence:.2%}"
                )

            if hazardous_probability is not None:

                st.metric(
                    "Probability of Hazardous Class",
                    f"{hazardous_probability:.2%}"
                )

            # --------------------------------------
            # INPUT DETAILS
            # --------------------------------------

            with st.expander(
                "View Asteroid Input Details"
            ):

                st.dataframe(
                    pd.DataFrame(
                        raw.items(),
                        columns=["Feature", "Input Value"]
                    ),
                    use_container_width=True
                )

            # --------------------------------------
            # PROCESSED FEATURES
            # --------------------------------------

            with st.expander(
                "Processed Features Sent to Model"
            ):

                st.dataframe(
                    X.T.rename(columns={0: "Value"}),
                    use_container_width=True
                )

            st.caption(
                "This is a machine learning prediction, "
                "not an official NASA hazard assessment."
            )

        except Exception as e:

            st.error(
                f"Prediction failed: {e}"
            )

# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.markdown(
    """
    <div style="text-align:center">
        <p>NASA Near-Earth Object Classification</p>
        <p>Machine Learning | Random Forest | Streamlit</p>
    </div>
    """,
    unsafe_allow_html=True
)
