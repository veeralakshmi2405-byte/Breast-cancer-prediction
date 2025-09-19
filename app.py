# app.py

import streamlit as st
import pickle
import numpy as np

# Load the model and scaler
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

def predict_breast_cancer(input_features):
    """
    input_features: list or array of 30 features in same order as training
    returns: prediction (0 or 1), and probabilities
    """
    arr = np.array(input_features).reshape(1, -1)
    arr_scaled = scaler.transform(arr)
    pred = model.predict(arr_scaled)[0]
    prob = model.predict_proba(arr_scaled)[0]
    return pred, prob

def main():
    st.title("Breast Cancer Prediction App")

    st.write("Enter the measurements of tumor cells to predict whether it's Benign or Malignant.")

    # Assume 30 features from breast cancer dataset
    feature_names = [
        "radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean",
        "compactness_mean", "concavity_mean", "concave_points_mean", "symmetry_mean",
        "fractal_dimension_mean",
        "radius_se", "texture_se", "perimeter_se", "area_se", "smoothness_se",
        "compactness_se", "concavity_se", "concave_points_se", "symmetry_se", "fractal_dimension_se",
        "radius_worst", "texture_worst", "perimeter_worst", "area_worst", "smoothness_worst",
        "compactness_worst", "concavity_worst", "concave_points_worst", "symmetry_worst",
        "fractal_dimension_worst"
    ]

    inputs = []
    for name in feature_names:
        val = st.number_input(f"{name}", value=0.0, format="%.5f")
        inputs.append(val)

    if st.button("Predict"):
        pred, prob = predict_breast_cancer(inputs)
        if pred == 1:
            st.success("Prediction: **Benign** ✅")
        else:
            st.error("Prediction: **Malignant** ⚠️")

        st.write(f"Probability Benign: {prob[1]:.4f}")
        st.write(f"Probability Malignant: {prob[0]:.4f}")

if __name__ == "__main__":
    main()
