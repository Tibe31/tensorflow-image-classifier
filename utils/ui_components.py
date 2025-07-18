import streamlit as st
import os
from PIL import Image

def display_image_gallery(directory, max_images=16):
    if not os.path.isdir(directory):
        st.warning(f"La cartella specificata non esiste: {directory}")
        return

    image_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                image_files.append(os.path.join(root, file))
            if len(image_files) >= max_images:
                break
        if len(image_files) >= max_images:
            break

    if not image_files:
        st.info(f"Nessuna immagine trovata nella cartella: {directory}")
        return

    st.write("### Anteprima Immagini")
    cols = st.columns(4) # 4 colonne per la galleria
    for i, img_path in enumerate(image_files):
        with cols[i % 4]:
            try:
                img = Image.open(img_path)
                st.image(img, caption=os.path.basename(img_path), use_container_width=True)
            except Exception as e:
                st.error(f"Errore nel caricare l'immagine {os.path.basename(img_path)}: {e}")