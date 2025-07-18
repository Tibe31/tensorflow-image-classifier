import streamlit as st
import os
from utils.config_loader import save_config
from utils.data_splitter import perform_auto_split
from utils.ui_components import display_image_gallery

def render(config_data):
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