import streamlit as st
import requests
from PIL import Image
import urllib3
import io
import json

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

API_BASE = "https://waste-prediction-api-production.up.railway.app"
LOGIN_URL = f"{API_BASE}/token"
REGISTER_URL = f"{API_BASE}/register"
PREDICT_URL = f"{API_BASE}/predict_all"

for key, default in {
    "token": None, "user": None, "image": None,
    "clear_inputs": False, "login_message": "", "register_message": ""
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

if st.session_state.clear_inputs:
    st.session_state.username = ""
    st.session_state.password = ""
    st.session_state.reg_username = ""
    st.session_state.reg_password = ""
    st.session_state.clear_inputs = False

def st_bin(message: str, waste_type: str = "unknown"):
    color_map = {
        "cardboard": "#3498db",  # niebieski – papier
        "paper":     "#3498db",
        "glass":     "#27ae60",  # zielony – szkło
        "metal":     "#f1c40f",  # żółty – metale/plastik
        "plastic":   "#f1c40f",
        "organic":   "#8B4513",  # brązowy – bio
        "e-waste":   "#2c3e50",  # ciemny – elektrośmieci
        "textiles":  "#9b59b6",  # fioletowy – tekstylia
        "medical":   "#e74c3c",  # czerwony – medyczne
        "wood":      "#d35400",  # pomarańczowy – drewno
    }
    bg_color = color_map.get(waste_type, "#636e72")  # domyślny szary

    st.markdown(f"""
    <div style="
        padding: 18px 24px;
        margin: 20px 0;
        border-radius: 16px;
        background: linear-gradient(135deg, {bg_color}22, {bg_color}44);
        border-left: 8px solid {bg_color};
        color: white;
        font-size: 18px;
        font-weight: 600;
        box-shadow: 0 6px 20px rgba(0,0,0,0.15);
        display: flex;
        align-items: center;
        gap: 14px;
    ">
        <span style="font-size: 22px;">Recycle</span>
        <div>
            Suggested disposal method:<br>
            {message}
        </div>
    </div>
    """, unsafe_allow_html=True)

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
            response = requests.post(LOGIN_URL, data={"username": username, "password": password}, verify=False)
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
                response = requests.post(REGISTER_URL, data={"username": reg_username, "password": reg_password}, verify=False)
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

main_col, hist_col = st.columns([4, 2])

with hist_col:
    if st.session_state.token:
        st.subheader("Your last predictions")

        try:
            headers = {"Authorization": f"Bearer {st.session_state.token}"}
            hist_response = requests.post(f"{API_BASE}/user_history", headers=headers, verify=False)
            hist_response.raise_for_status()

            history = hist_response.json().get("history_list", [])

            if len(history) == 0:
                st.info("No history yet.")
            else:
                for raw_item in history:
                    item = json.loads(raw_item)  # ← parsowanie JSON STRINGU

                    with st.container(border=True):
                        st.write(f"**Prediction:** {item['trash_prediction']}")
                        st.write(f"**Date:** {item['created_at']}")
                        if item["photo_link"] != "upload failed":
                            st.image(item["photo_link"], width=140)
                        else:
                            st.write("Image upload failed.")

        except Exception as e:
            st.error(f"Could not load history: {e}")

with main_col:

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
                    if preds[0] == 'not trash':
                        st.success("Not trash")
                    else:
                        st.success(f"Top prediction: {preds[0].title()}")

                        disposal_texts = {
                            "cardboard": "Blue bin - paper and cardboard",
                            "paper":     "Blue bin - paper and cardboard",
                            "glass":     "Green bin - glass",
                            "metal":     "Yellow bin - plastic and metal waste",
                            "plastic":   "Yellow bin - plastic and metal waste",
                            "organic":   "Brown bin - biodegradable waste",
                            "e-waste":   "Special collection point for electronic waste",
                            "textiles":  "Textile container or textile collection point",
                            "medical":   "Special collection point for medical waste",
                            "wood":      "Bulk waste or wood recycling point",
                        }
                        text = disposal_texts.get(preds[0], "Check local waste disposal rules")
                        st_bin(text, waste_type=preds[0])

                else:
                    st.warning("No predictions returned.")
            except Exception as e:
                st.error(f"Error: {e}")