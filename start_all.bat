@echo off
echo ============================================
echo   Sentiment Analysis ML Platform
echo ============================================
echo.
echo Iniciando todos os serviços...
echo.

echo 1. Iniciando MLflow Server...
start "MLflow Server" cmd /k "cd mlflow-tracking && python start_mlflow.py"
timeout /t 5

echo 2. Iniciando FastAPI Backend...
start "FastAPI Backend" cmd /k "cd ml-api && uvicorn app.main:app --reload --host 0.0.0.0 --port 8000"
timeout /t 5

echo 3. Iniciando Streamlit Frontend...
start "Streamlit Frontend" cmd /k "cd streamlit-frontend && streamlit run app.py --server.port 8501"

echo.
echo ============================================
echo   Serviços Iniciados:
echo   - MLflow:    http://localhost:5000
echo   - API:       http://localhost:8000
echo   - Frontend:  http://localhost:8501
echo ============================================
echo.
echo Pressione qualquer tecla para fechar...
pause