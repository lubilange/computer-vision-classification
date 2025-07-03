import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import requests
import pathlib
import os, uuid
from io import BytesIO

# ──────────────────────────────────────────────────────────────
# 1. Téléchargement automatique du modèle depuis Hugging Face
# ──────────────────────────────────────────────────────────────
MODEL_URL  = "https://huggingface.co/Angeusafico/cv/resolve/main/modele_ia.h5"
MODEL_PATH = pathlib.Path("modele_ia.h5")

@st.cache_resource  # téléchargé et chargé une seule fois
def load_model():
    if not MODEL_PATH.exists():
        with st.spinner("📥 Téléchargement du modèle (~267 Mo)…"):
            r = requests.get(MODEL_URL, stream=True)
            r.raise_for_status()
            with open(MODEL_PATH, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        st.success("Modèle téléchargé ✅")
    return tf.keras.models.load_model(MODEL_PATH, compile=False)

model = load_model()
classes = ['fertile', 'non_fertile']

# ──────────────────────────────────────────────────────────────
# 2. Fonctions utilitaires
# ──────────────────────────────────────────────────────────────
def predict_image(img: Image.Image) -> str:
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.reshape(img_array, [-1, 224, 224, 3])
    prediction = model.predict(img_array)
    return classes[int(np.argmax(prediction))]

# ──────────────────────────────────────────────────────────────
# 3. Interface Streamlit
# ──────────────────────────────────────────────────────────────
st.title("Classification d'image : Fertile / Non Fertile")
st.write("Chargez une ou plusieurs images et laissez l’IA vous aider")

uploaded_files = st.file_uploader(
    "Choisir une ou plusieurs images",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

if uploaded_files:
    for uploaded_file in uploaded_files:
        image_data = uploaded_file.read()
        image = Image.open(BytesIO(image_data)).convert("RGB")

        st.image(image, caption=uploaded_file.name, use_column_width=True)

        if st.button(f"Prédire « {uploaded_file.name} »"):
            result = predict_image(image)
            st.success(f"Résultat : **{result.upper()}**")

            # Crée les dossiers s'ils n'existent pas et sauvegarde l'image
            os.makedirs("fertile", exist_ok=True)
            os.makedirs("non_fertile", exist_ok=True)

            unique_filename = f"{uuid.uuid4().hex}.jpg"
            save_path = os.path.join(result, unique_filename)
            image.save(save_path)
            st.info(f"📁 Image sauvegardée dans `{result}/{unique_filename}`")
