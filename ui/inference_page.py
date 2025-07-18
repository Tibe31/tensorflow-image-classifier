import streamlit as st
import os
from utils.inference import predict
from PIL import Image

def render(config_data):
    st.header("3. Inferenza")

    # --- Model Selection ---
    model_dir = config_data['checkpoint_filepath']
    if not os.path.isdir(model_dir):
        st.warning(f"La cartella dei modelli specificata ({model_dir}) non esiste. Esegui prima il training.")
        return

    models = [d for d in os.listdir(model_dir) if os.path.isdir(os.path.join(model_dir, d))]
    if not models:
        st.warning(f"Nessun modello (cartella) trovato nella directory: {model_dir}. Esegui prima il training.")
        return

    models.sort(key=lambda d: os.path.getmtime(os.path.join(model_dir, d)), reverse=True)
    selected_model_name = st.selectbox("Seleziona un modello", models)
    model_path = os.path.join(model_dir, selected_model_name)

    # --- Inference Mode ---
    inference_mode = st.radio("Scegli la modalità di inferenza:", ("File Singolo", "Cartella"))

    if inference_mode == "File Singolo":
        uploaded_file = st.file_uploader("Carica un'immagine...", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            image_to_show = Image.open(uploaded_file)
            st.image(image_to_show, caption='Immagine Caricata', use_container_width=True)
            if st.button("Esegui Inferenza"):
                with st.spinner("Inferenza in corso..."):
                    prediction = predict(uploaded_file, model_path, config_data['input_shape'])
                    st.success("Inferenza completata!")
                    st.metric(label="Punteggio di predizione", value=f"{prediction[0][0]:.4f}")
                    if prediction[0][0] > 0.5:
                        st.success("**Risultato:** L'immagine appartiene alla classe 1")
                    else:
                        st.error("**Risultato:** L'immagine appartiene alla classe 0")

    elif inference_mode == "Cartella":
        folder_path = st.text_input("Inserisci il percorso della cartella")
        st.info("Suggerimento: Apri la cartella in Esplora File, clicca sulla barra dell'indirizzo in alto, copia il percorso e incollalo qui.")

        if st.button("Esegui Inferenza su Cartella"):
            if not os.path.isdir(folder_path):
                st.error("Il percorso inserito non è una cartella valida.")
                return

            image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if not image_files:
                st.warning("Nessuna immagine trovata nella cartella specificata.")
                return

            st.subheader("Risultati Inferenza (in tempo reale)")
            progress_bar = st.progress(0)

            with st.spinner(f"Inferenza in corso su {len(image_files)} immagini..."):
                for i, filename in enumerate(image_files):
                    img_path = os.path.join(folder_path, filename)
                    try:
                        prediction = predict(img_path, model_path, config_data['input_shape'])
                        score = prediction[0][0]
                        predicted_class = 1 if score > 0.5 else 0

                        # Mostra i risultati immediatamente
                        col1, col2, col3 = st.columns([1, 3, 1])
                        with col1:
                            st.image(img_path, width=100, caption=filename)
                        with col2:
                            st.metric(label="Classe Predetta", value=predicted_class)
                        with col3:
                            st.metric(label="Punteggio", value=f"{score:.4f}")
                        st.divider()

                    except Exception as e:
                        st.error(f"Errore durante l'inferenza su {filename}: {e}")
                    
                    # Aggiorna la barra di avanzamento
                    progress_bar.progress((i + 1) / len(image_files))

            st.success("Inferenza completata!")