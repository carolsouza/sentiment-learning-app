@echo off
echo ============================================
echo   Instalando Dependências do Projeto
echo ============================================
echo.

echo 1. Instalando dependências do MLflow...
cd mlflow-tracking
pip install -r requirements.txt
cd ..
echo.

echo 2. Instalando dependências da ML API...
cd ml-api
pip install -r requirements.txt
cd ..
echo.

echo 3. Instalando dependências do Streamlit Frontend...
cd streamlit-frontend
pip install -r requirements.txt
cd ..
echo.

echo ============================================
echo   Instalação Concluída!
echo ============================================
echo.
echo Para iniciar o projeto, execute: start_all.bat
echo.
pause