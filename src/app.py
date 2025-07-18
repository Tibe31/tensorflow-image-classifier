import streamlit as st
from streamlit_option_menu import option_menu
from src.utils.config_loader import load_config
from src.ui import config_page, training_page, inference_page

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
        config_page.render(config_data)
    elif page == "Training":
        training_page.render(config_data)
    elif page == "Inferenza":
        inference_page.render(config_data)


if __name__ == "__main__":
    main()