import streamlit as st
import requests
from PIL import Image
import urllib3
import io

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

API_BASE = "https://waste-prediction-api-production.up.railway.app"
LOGIN_URL = f"{API_BASE}/token"
REGISTER_URL = f"{API_BASE}/register"
PREDICT_URL = f"{API_BASE}/predict_all"


for key, default in {
    "token": None,
    "user": None,
    "image": None,
    "clear_inputs": False,
    "login_message": "",
    "register_message": "",
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

if st.session_state.clear_inputs:
    st.session_state.username = ""
    st.session_state.password = ""
    st.session_state.reg_username = ""
    st.session_state.reg_password = ""
    st.session_state.clear_inputs = False


col1, col2 = st.columns([6, 1])
with col2:
    if st.session_state.token:
        st.write(f"Logged as: {st.session_state.user}")
        if st.button("Logout"):
            st.session_state.token = None
            st.session_state.user = None
            st.session_state.clear_inputs = True
            st.session_state.login_message = "Logged out successfully!"
            st.rerun()


tab1, tab2 = st.sidebar.tabs(["Login", "Register"])

with tab1:
    st.header("Login")
    username = st.text_input("Username", key="username")
    password = st.text_input("Password", type="password", key="password")


    if st.session_state.login_message:
        if "successful" in st.session_state.login_message.lower():
            st.success(st.session_state.login_message)
        else:
            st.error(st.session_state.login_message)

    if st.button("Login"):
        if not username or not password:
            st.session_state.login_message = "Enter username and password!"
            st.rerun()
        try:
            response = requests.post(
                LOGIN_URL,
                data={"username": username, "password": password},
                verify=False,
            )
            response.raise_for_status()
            st.session_state.token = response.json()["access_token"]
            st.session_state.user = username
            st.session_state.clear_inputs = True
            st.session_state.login_message = "Login successful!"
            st.rerun()
        except Exception:
            st.session_state.login_message = "Login error — wrong credentials?"
            st.rerun()

with tab2:
    st.header("Register")

    if st.session_state.get("clear_register_inputs", False):
        st.session_state.reg_username = ""
        st.session_state.reg_password = ""
        st.session_state.clear_register_inputs = False

    reg_username = st.text_input("New username", key="reg_username")
    reg_password = st.text_input("New password", type="password", key="reg_password")



    if st.session_state.register_message:
        if "created" in st.session_state.register_message.lower():
            st.success(st.session_state.register_message)
        else:
            st.error(st.session_state.register_message)

    if st.button("Create Account"):
        if not reg_username or not reg_password:
            st.session_state.register_message = "Enter username and password!"
            st.rerun()
        else:
            try:
                response = requests.post(
                    REGISTER_URL,
                    data={"username": reg_username, "password": reg_password},
                    verify=False
                )
                if response.status_code == 200:
                    st.session_state.register_message = "Account created! You can now log in."
                    st.session_state.clear_register_inputs = True
                    st.rerun()
                else:
                    detail = response.json().get("detail", "Registration failed")
                    st.session_state.register_message = detail
                    st.rerun()
            except Exception as e:
                st.session_state.register_message = f"Error: {e}"
                st.rerun()


st.title("Waste Prediction App")
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
