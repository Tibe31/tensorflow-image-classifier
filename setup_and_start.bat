@echo off
:: Questo script imposta l'ambiente Conda 'yolov8' e avvia l'applicazione.

SET ENV_NAME=yolov8

:: 1. Controlla se l'ambiente esiste già
conda env list | findstr /B /C:"%ENV_NAME% " >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    ECHO Ambiente Conda '%ENV_NAME%' non trovato. Creazione in corso...
    
    :: Crea un nuovo ambiente con il nome specificato e Python 3.9
    conda create --name %ENV_NAME% python=3.9 -y
    IF %ERRORLEVEL% NEQ 0 (
        ECHO ERRORE: Creazione dell'ambiente Conda fallita.
        PAUSE
        EXIT /B 1
    )

    ECHO Attivazione del nuovo ambiente...
    CALL conda activate %ENV_NAME%
    IF %ERRORLEVEL% NEQ 0 (
        ECHO ERRORE: Attivazione dell'ambiente fallita.
        PAUSE
        EXIT /B 1
    )

    ECHO Installazione delle dipendenze da requirements.txt...
    pip install -r requirements.txt
    IF %ERRORLEVEL% NEQ 0 (
        ECHO ERRORE: Installazione delle dipendenze fallita.
        PAUSE
        EXIT /B 1
    )

) ELSE (
    ECHO Ambiente Conda '%ENV_NAME%' trovato. Attivazione...
    CALL conda activate %ENV_NAME%
    IF %ERRORLEVEL% NEQ 0 (
        ECHO ERRORE: Attivazione dell'ambiente esistente fallita.
        PAUSE
        EXIT /B 1
    )
)

ECHO Avvio dell'applicazione Streamlit...
streamlit run src/app.py

ECHO Chiusura dello script.
PAUSE