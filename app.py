import streamlit as st
import cv2
import numpy as np
import joblib
import json
from skimage.feature import local_binary_pattern
from PIL import Image

st.set_page_config(page_title="Crowd Density Classification")

st.title("Crowd Density Classification")
st.write("Upload an image to predict crowd density level.")

# load class names
with open("class_names.json") as f:
    class_names = json.load(f)

# load trained classifier (HOG-based)
clf = joblib.load("crowd_classifier.pkl")

uploaded_file = st.file_uploader(
    "Upload image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", width=600)

    # --- preprocessing (HARUS sama dengan training) ---
    img = np.array(image)
    img = cv2.resize(img, (124, 124))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    LBP_RADIUS = 1
    LBP_POINTS = 8 * LBP_RADIUS
    LBP_METHOD = "uniform"

    # --- LBP feature extraction ---
    lbp = local_binary_pattern(
    img,
    P=LBP_POINTS,
    R=LBP_RADIUS,
    method=LBP_METHOD
)

# histogram LBP (uniform -> P + 2 bins)
    hist, _ = np.histogram(
    lbp.ravel(),
    bins=np.arange(0, LBP_POINTS + 3),
    range=(0, LBP_POINTS + 2),
    density=True
)
    features = hist.reshape(1, -1)

    pred = clf.predict(features)[0]
    pred_label = class_names[pred]

    st.subheader("Prediction Result")
    st.success(f"Crowd Density: **{pred_label.upper()}**")