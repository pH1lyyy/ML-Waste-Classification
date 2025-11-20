import streamlit as st
import requests
from PIL import Image
import urllib3
import io

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

API_BASE = "https://waste-prediction-api-production.up.railway.app"
LOGIN_URL = f"{API_BASE}/token"
PREDICT_URL = f"{API_BASE}/predict_all"


if "token" not in st.session_state:
    st.session_state.token = None
if "image" not in st.session_state:
    st.session_state.image = None



st.sidebar.header("Login")
username = st.sidebar.text_input("Username")
password = st.sidebar.text_input("Password", type="password")

if st.sidebar.button("Login"):
    if not username or not password:
        st.sidebar.error("Enter username and password!")
    else:
        try:
            response = requests.post(
                LOGIN_URL,
                data={"username": username, "password": password},
                verify=False,
            )
            response.raise_for_status()
            st.session_state.token = response.json()["access_token"]
            st.sidebar.success("Login successful!")
        except Exception as e:
            st.sidebar.error(f"Login error: {e}")



st.title("Waste Prediction - Demo")

uploaded_file = st.file_uploader("Choose a photo", type=["jpg", "jpeg", "png", "bmp"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.session_state.image = image
    st.image(image, caption="Selected image", width=420)


if st.button("Analyze photo"):
    if not st.session_state.image:
        st.error("Please upload a photo first!")
    elif not st.session_state.token:
        st.error("Please log in first!")
    else:
        try:
            img_bytes = io.BytesIO()
            st.session_state.image.save(img_bytes, format="PNG")
            img_bytes.seek(0)

            files = {"img": ("image.png", img_bytes, "image/png")}
            headers = {"Authorization": f"Bearer {st.session_state.token}"}
            response = requests.post(PREDICT_URL, files=files, headers=headers, verify=False)
            response.raise_for_status()
            answer = response.json()

            preds = answer.get("all_predictions", [])
            if preds:
                st.success(f"Top prediction: {preds[0]}")
                st.info(f"All predictions: {', '.join(preds)}")
            else:
                st.warning("No predictions returned.")

        except Exception as e:
            st.error(f"Error: {e}")
