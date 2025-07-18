import streamlit as st
import yaml
from utils.config_loader import Config
from utils.data_splitter import perform_auto_split
import subprocess
import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from utils.utils import show_augmentations
from PIL import Image
from streamlit_option_menu import option_menu

# Funzione per caricare la configurazione
def load_config():
    return Config(path='config.yaml')

# Funzione per salvare la configurazione
def save_config(config_data):
    # Ensure input_shape is a list before saving
    if 'input_shape' in config_data and isinstance(config_data['input_shape'], tuple):
        config_data['input_shape'] = list(config_data['input_shape'])
    with open('config.yaml', 'w') as f:
        yaml.dump(config_data, f)

# Funzione per l'inferenza
def predict(image_input, model_path, input_shape):
    model = tf.keras.models.load_model(model_path)
    
    if isinstance(image_input, str): # If it's a file path
        img = image.load_img(image_input, target_size=(input_shape[0], input_shape[1]))
    else: # Assume it's a Streamlit UploadedFile object
        pil_img = Image.open(image_input).convert('RGB') # Ensure 3 channels
        img = pil_img.resize((input_shape[0], input_shape[1])) # Resize PIL image

    img_array = image.img_to_array(img) # Convert PIL Image to numpy array
    img_array = np.expand_dims(img_array, axis=0) # Add batch dimension
    
    # Normalization is handled by the model/training pipeline, so we remove it here.
    # img_array /= 255.0 
    
    prediction = model.predict(img_array)
    return prediction

# Funzione per mostrare una galleria di immagini da una directory
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

# Pagina Principale
def main():
    st.title("Workflow Classificazione Immagini")

    # Carica la configurazione
    config = load_config()
    config_data = config.cfg

    # Inizializza lo stato della sessione per il training
    if 'training_process' not in st.session_state:
        st.session_state.training_process = None
    if 'training_running' not in st.session_state:
        st.session_state.training_running = False

    with st.sidebar:
        page = option_menu("Navigazione", 
                           ["Configurazione", "Training", "Inferenza"],
                           icons=['gear', 'activity', 'search'], 
                           menu_icon="cast", default_index=0)

    if page == "Configurazione":
        st.header("1. Configurazione e Preparazione Dati")

        # Mostra e modifica la configurazione
        st.subheader("Modifica la configurazione")
        
        # Sezione Auto-Split
        st.write("### Auto-Split del Dataset")
        config_data['auto_split']['enabled'] = st.checkbox("Abilita Auto-Split", config_data['auto_split']['enabled'])
        source_dir = st.text_input("Cartella di origine", config_data['auto_split']['source_dir'])
        if not os.path.isdir(source_dir):
            st.warning(f"La cartella di origine specificata non esiste o non è una directory valida: {source_dir}")
        config_data['auto_split']['source_dir'] = source_dir

        # Mostra la galleria di immagini dalla cartella di origine
        display_image_gallery(source_dir, max_images=4)

        output_dir = st.text_input("Cartella di output", config_data['auto_split']['output_dir'])
        if not os.path.isdir(output_dir):
            st.warning(f"La cartella di output specificata non esiste o non è una directory valida: {output_dir}")
        config_data['auto_split']['output_dir'] = output_dir

        config_data['auto_split']['val_split'] = st.slider("Percentuale di validazione", 0.0, 1.0, float(config_data['auto_split']['val_split']))
        config_data['auto_split']['test_split'] = st.slider("Percentuale di test", 0.0, 1.0, float(config_data['auto_split']['test_split']))

        # Pulsante per salvare la configurazione
        if st.button("Salva Configurazione"):
            save_config(config_data)
            st.success("Configurazione salvata con successo!")

        # Pulsante per eseguire lo split
        if st.button("Esegui Split del Dataset"):
            if config_data['auto_split']['enabled']:
                with st.spinner("Esecuzione dello split in corso..."):
                    train_dir_actual, val_dir_actual, test_dir_actual = perform_auto_split(config_data)
                    st.success(f"Split completato con successo!")
                    st.info(f"Train dir: {train_dir_actual}")
                    st.info(f"Validation dir: {val_dir_actual}")
                    st.info(f"Test dir: {test_dir_actual}")

                    # Update config_data with the actual paths
                    config_data['train_dir'] = train_dir_actual
                    config_data['val_dir'] = val_dir_actual
                    config_data['test_path'] = test_dir_actual # Assuming test_path should be test_dir

                    # Save the updated config
                    save_config(config_data)
                    st.success("Percorsi di training/validazione/test aggiornati e salvati nella configurazione.")
                    st.rerun() # Rerun to update the text inputs in the Training tab
            else:
                st.warning("L'auto-split non è abilitato nella configurazione.")


    elif page == "Training":
        st.header("2. Training del Modello")

        # Parametri da escludere dalla modifica diretta tramite loop generale
        excluded_params = [
            'auto_split', 'augmentation', 'label_dictionary', 'show_augmentations',
            'train_dir', 'val_dir', 'test_path', 'use_pretrained_model',
            'pretrained_model_path', 'checkpoint_filepath', 'monitor_metric', 'mode'
        ]

        tab1, tab2, tab3, tab4 = st.tabs(["Configurazione Generale", "Iperparametri", "Data Augmentation", "Avvia/Monitora Training"])

        with tab1:
            st.subheader("Configurazione Generale")

            # Percorsi delle directory
            train_dir = st.text_input("Cartella Training", config_data['train_dir'])
            if not os.path.isdir(train_dir):
                st.warning(f"La cartella di training specificata non esiste o non è una directory valida: {train_dir}")
            config_data['train_dir'] = train_dir

            val_dir = st.text_input("Cartella Validazione", config_data['val_dir'])
            if not os.path.isdir(val_dir):
                st.warning(f"La cartella di validazione specificata non esiste o non è una directory valida: {val_dir}")
            config_data['val_dir'] = val_dir

            test_path = st.text_input("Cartella Test", config_data['test_path'])
            if not os.path.isdir(test_path):
                st.warning(f"La cartella di test specificata non esiste o non è una directory valida: {test_path}")
            config_data['test_path'] = test_path

            # Modello Pre-addestrato
            config_data['use_pretrained_model'] = st.checkbox("Usa Modello Pre-addestrato", config_data['use_pretrained_model'])
            if config_data['use_pretrained_model']:
                pretrained_model_path = st.text_input("Percorso Modello Pre-addestrato", config_data['pretrained_model_path'])
                if not os.path.isfile(pretrained_model_path):
                    st.warning(f"Il modello pre-addestrato specificato non esiste o non è un file valido: {pretrained_model_path}")
                config_data['pretrained_model_path'] = pretrained_model_path
            else:
                config_data['pretrained_model_path'] = "" # Reset if not used

            # Checkpoint e Metriche
            checkpoint_filepath = st.text_input("Percorso Checkpoint", config_data['checkpoint_filepath'])
            if not os.path.isdir(os.path.dirname(checkpoint_filepath)):
                st.warning(f"La directory per il checkpoint specificata non esiste o non è una directory valida: {os.path.dirname(checkpoint_filepath)}")
            config_data['checkpoint_filepath'] = checkpoint_filepath
            config_data['monitor_metric'] = st.text_input("Metrica da Monitorare", config_data['monitor_metric'])
            config_data['mode'] = st.selectbox("Modalità Monitoraggio", ['min', 'max'], index=['min', 'max'].index(config_data['mode']))

        with tab2:
            st.subheader("Altri Iperparametri di Training")
            for key, value in config_data.items():
                if key not in excluded_params and not isinstance(value, dict):
                    if isinstance(value, bool):
                        config_data[key] = st.checkbox(key, value)
                    elif isinstance(value, int):
                        config_data[key] = st.number_input(key, value=value)
                    elif isinstance(value, float):
                        config_data[key] = st.number_input(key, value=value, format="%f")
                    else:
                        config_data[key] = st.text_input(key, value)

        with tab3:
            st.subheader("Data Augmentation")
            config_data['show_augmentations'] = st.checkbox("Mostra Esempi di Augmentation", config_data['show_augmentations'])

            if 'augmentation' in config_data:
                for aug_type, aug_params in config_data['augmentation'].items():
                    with st.expander(f"Augmentation - {aug_type.capitalize()}"):
                        for key, value in aug_params.items():
                            if isinstance(value, bool):
                                config_data['augmentation'][aug_type][key] = st.checkbox(f"{key.replace('_', ' ').capitalize()}", value, key=f"aug_{aug_type}_{key}_checkbox")
                            elif isinstance(value, int):
                                config_data['augmentation'][aug_type][key] = st.number_input(f"{key.replace('_', ' ').capitalize()}", value=value, key=f"aug_{aug_type}_{key}_int")
                            elif isinstance(value, float):
                                config_data['augmentation'][aug_type][key] = st.number_input(f"{key.replace('_', ' ').capitalize()}", value=value, format="%f", key=f"aug_{aug_type}_{key}_float")
                            elif isinstance(value, list) and len(value) == 2 and all(isinstance(x, (int, float)) for x in value):
                                # Gestione specifica per le liste (es. brightness_range)
                                slider_tuple = st.slider(f"{key.replace('_', ' ').capitalize()}", float(value[0]), float(value[1]), (float(value[0]), float(value[1])), key=f"aug_{aug_type}_{key}_slider")
                                config_data['augmentation'][aug_type][key] = list(slider_tuple)
                            else:
                                config_data['augmentation'][aug_type][key] = st.text_input(f"{key.replace('_', ' ').capitalize()}", value, key=f"aug_{aug_type}_{key}_text")

        with tab4:
            st.subheader("Controllo Training")
            # Pulsante Avvia/Stop Training
            if st.session_state.training_running:
                if st.button("Stop Training"):
                    if st.session_state.training_process:
                        st.session_state.training_process.terminate()  # Termina il processo
                        st.session_state.training_process.wait()  # Attendi la terminazione
                        st.session_state.training_process = None
                    st.session_state.training_running = False
                    st.warning("Training interrotto!")
                    st.rerun() # Ricarica la pagina per aggiornare lo stato del pulsante
            else:
                if st.button("Avvia Training"):
                    st.info("Avvio del processo di training...")
                    save_config(config_data) # Salva la configurazione prima di avviare il training
                    st.success("Configurazione salvata su disco prima dell'avvio del training.")

                    # Mostra le augmentation se abilitato, PRIMA di avviare train.py
                    if config_data['show_augmentations']:
                        train_data_dir = config_data['train_dir']
                        if not os.path.exists(train_data_dir):
                            st.error(f"Errore: La cartella di training specificata ({train_data_dir}) non esiste. Esegui prima lo split del dataset o correggi il percorso in config.yaml.")
                        else:
                            try:
                                fig = show_augmentations(
                                    train_dir=train_data_dir,
                                    batch_size=config_data['batch_size'],
                                    input_shape=config_data['input_shape'],
                                    augmentation_parameters=config_data['augmentation']['train'],
                                    num_classes=config_data['classes']
                                )
                                st.pyplot(fig)
                                st.success("Esempi di augmentation visualizzati.")
                            except Exception as e:
                                st.error(f"Errore durante la visualizzazione delle augmentation: {e}. Assicurati che la cartella di training contenga immagini valide.")

                    st.session_state.training_running = True
                    # Avvia il processo in background e memorizzalo nello stato della sessione
                    st.session_state.training_process = subprocess.Popen(
                        ['python', 'train.py'],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        bufsize=1,
                        universal_newlines=True
                    )
                    st.rerun() # Ricarica la pagina per mostrare il pulsante Stop

            # Mostra l'output del training se in esecuzione
            if st.session_state.training_running and st.session_state.training_process:
                st.subheader("Output Training")
                output_container = st.empty()
                output = ""
                augmentation_image_path = None
                augmentation_image_displayed = False # Flag per controllare se l'immagine è già stata mostrata

                while True:
                    line = st.session_state.training_process.stdout.readline()
                    if not line:
                        break
                    output += line
                    output_container.code(output)

                    # Cerca il percorso dell'immagine di augmentation nell'output
                    if "AUGMENTATION_IMAGE_PATH:" in line and not augmentation_image_displayed:
                        augmentation_image_path = line.split("AUGMENTATION_IMAGE_PATH:")[1].strip()
                        if augmentation_image_path and os.path.exists(augmentation_image_path):
                            st.subheader("Esempi di Augmentation")
                            st.image(augmentation_image_path, caption="Immagini Aumentate")
                            os.remove(augmentation_image_path) # Pulisci il file temporaneo subito dopo la visualizzazione
                            augmentation_image_displayed = True # Imposta il flag a True per non visualizzare più volte
                
                # Controlla se il processo è terminato
                if st.session_state.training_process.poll() is not None:
                    st.session_state.training_running = False
                    st.session_state.training_process = None
                    st.success("Training completato!")
                    st.rerun() # Ricarica la pagina per aggiornare lo stato del pulsante


    elif page == "Inferenza":
        st.header("3. Inferenza")
        
        # Seleziona il modello
        model_dir = config_data['checkpoint_filepath']
        if os.path.isdir(model_dir):
            # List subdirectories, which are the saved models
            models = [d for d in os.listdir(model_dir) if os.path.isdir(os.path.join(model_dir, d))]
            if not models:
                st.warning(f"Nessun modello (cartella) trovato nella directory: {model_dir}. Esegui prima il training.")
                return
            
            # Sort models by modification time (most recent first)
            models.sort(key=lambda d: os.path.getmtime(os.path.join(model_dir, d)), reverse=True)
            
            selected_model = st.selectbox("Seleziona un modello", models)
            model_path = os.path.join(model_dir, selected_model)
        else:
            st.warning(f"La cartella dei modelli specificata ({model_dir}) non esiste. Esegui prima il training.")
            return

        # Carica immagine
        uploaded_file = st.file_uploader("Carica un'immagine...", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            # Mostra l'immagine
            image_to_show = Image.open(uploaded_file)
            st.image(image_to_show, caption='Immagine Caricata', use_container_width=True)

            # Esegui l'inferenza
            if st.button("Esegui Inferenza"):
                with st.spinner("Inferenza in corso..."):
                    # Passa direttamente l'oggetto uploaded_file alla funzione predict
                    prediction = predict(uploaded_file, model_path, config_data['input_shape'])
                    st.success("Inferenza completata!")
                    st.metric(label="Punteggio di predizione", value=f"{prediction[0][0]:.4f}")
                    if prediction[0][0] > 0.5:
                        st.success("**Risultato:** L'immagine appartiene alla classe 1")
                    else:
                        st.error("**Risultato:** L'immagine appartiene alla classe 0")


if __name__ == "__main__":
    main()