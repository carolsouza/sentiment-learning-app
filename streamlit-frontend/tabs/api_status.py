import streamlit as st
import requests


def render_api_status_tab(api_client, api_status):
    """Renderiza a tab de API Status"""
    st.header("📋 Status da API e Sistema")

    col1, col2 = st.columns(2)

    with col1:
        _render_services_status(api_status)

    with col2:
        _render_system_information()


def _render_services_status(api_status):
    """Renderiza o status dos serviços"""
    st.subheader("🔧 Status dos Serviços")

    # Status da ML API
    if api_status:
        st.success("✅ ML API: Online (http://localhost:8000)")
    else:
        st.error("❌ ML API: Offline")
        st.write("Para iniciar a API:")
        st.code("cd ml-api && uvicorn app.main:app --reload", language="bash")

    # Status do MLflow (verificar se está rodando)
    mlflow_status = _check_mlflow_status()

    if mlflow_status:
        st.success("✅ MLflow: Online (http://localhost:5000)")
    else:
        st.warning("⚠️ MLflow: Offline")
        st.write("Para iniciar o MLflow:")
        st.code("cd mlflow-tracking && python start_mlflow.py", language="bash")

    # Botão de refresh
    if st.button("🔄 Atualizar Status"):
        st.rerun()


def _check_mlflow_status():
    """Verifica o status do MLflow"""
    try:
        response = requests.get("http://localhost:5000", timeout=5)
        return response.status_code == 200
    except:
        return False


def _render_system_information():
    """Renderiza informações do sistema"""
    st.subheader("📊 Informações do Sistema")

    # Informações gerais
    st.info("**Componentes do Sistema:**")
    st.write("• **Streamlit Frontend**: Interface web principal")
    st.write("• **FastAPI Backend**: API para treinamento de modelos")
    st.write("• **MLflow**: Tracking de experimentos e modelos")
    st.write("• **Naive Bayes + TF-IDF**: Modelo baseline")

    # Links úteis
    st.subheader("🔗 Links Úteis")

    # Verifica novamente os status para os links
    api_status = _check_api_status()
    mlflow_status = _check_mlflow_status()

    if api_status:
        st.markdown("• [API Docs (Swagger)](http://localhost:8000/docs)")
        st.markdown("• [API Health Check](http://localhost:8000/health)")
    else:
        st.write("• API Docs: Indisponível (API offline)")

    if mlflow_status:
        st.markdown("• [MLflow Dashboard](http://localhost:5000)")
    else:
        st.write("• MLflow Dashboard: Indisponível (MLflow offline)")

    # Comandos úteis
    st.subheader("🛠️ Comandos Úteis")

    with st.expander("Scripts de Inicialização"):
        st.code("""
# Iniciar todos os serviços (Windows)
start_all.bat

# Ou individualmente:
cd mlflow-tracking && python start_mlflow.py
cd ml-api && uvicorn app.main:app --reload
cd streamlit-frontend && streamlit run app.py
        """, language="bash")

    with st.expander("Instalação de Dependências"):
        st.code("""
# Instalar todas as dependências (Windows)
install_dependencies.bat

# Ou individualmente:
cd ml-api && pip install -r requirements.txt
cd streamlit-frontend && pip install -r requirements.txt
cd mlflow-tracking && pip install -r requirements.txt
        """, language="bash")

    # Informações de debug
    if st.checkbox("🔍 Informações de Debug"):
        _render_debug_information()


def _check_api_status():
    """Verifica o status da API"""
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        return response.status_code == 200
    except:
        return False


def _render_debug_information():
    """Renderiza informações de debug"""
    st.subheader("🐛 Debug Information")

    # Session state info
    st.write("**Session State Keys:**")
    session_keys = list(st.session_state.keys())
    st.write(f"Total keys: {len(session_keys)}")

    for key in session_keys:
        if key in ['dataset_info', 'dataset_stats', 'last_training_result']:
            st.write(f"• {key}: ✅ Disponível")
        else:
            st.write(f"• {key}: {type(st.session_state[key]).__name__}")

    # API connectivity test
    st.write("**Testes de Conectividade:**")

    try:
        # Teste API
        api_response = requests.get("http://localhost:8000/health", timeout=2)
        st.write(f"• API Response: {api_response.status_code}")
    except Exception as e:
        st.write(f"• API Error: {str(e)}")

    try:
        # Teste MLflow
        mlflow_response = requests.get("http://localhost:5000", timeout=2)
        st.write(f"• MLflow Response: {mlflow_response.status_code}")
    except Exception as e:
        st.write(f"• MLflow Error: {str(e)}")

    # Clear cache options
    st.write("**Cache Management:**")
    col1, col2 = st.columns(2)

    with col1:
        if st.button("🗑️ Clear Data Cache"):
            st.cache_data.clear()
            st.success("Data cache cleared!")

    with col2:
        if st.button("🗑️ Clear Resource Cache"):
            st.cache_resource.clear()
            st.success("Resource cache cleared!")