import streamlit as st
import os
import subprocess
from utils.config_loader import save_config
from utils.utils import show_augmentations

def render(config_data):
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