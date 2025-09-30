import streamlit as st
from api_client import DeepLearningAPIClient
from tabs.baseline_training import render_baseline_training_tab
from tabs.production_analysis import render_production_analysis_tab
from tabs.model_testing import render_model_testing_tab
from tabs.architecture_diagram import render_architecture_diagram_tab
from tabs.api_status import render_api_status_tab

# Configuração da página
st.set_page_config(
    page_title="Sentiment Analysis - ML Training Platform",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configuração para permitir uploads de até 500MB
try:
    st._config.set_option("server.maxUploadSize", 500)
except:
    # Fallback se a configuração falhar
    pass

# Inicializa cliente da API Deep Learning
api_client = DeepLearningAPIClient()

# Debug: Verifica se o método existe
if not hasattr(api_client, 'predict_baseline'):
    st.error("⚠️ Método predict_baseline não encontrado. Recarregue a página.")

# CSS customizado
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .experiment-card {
        border: 1px solid #ddd;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 0.5rem 0;
        background-color: #fafafa;
    }
</style>
""", unsafe_allow_html=True)

# Cabeçalho principal
st.markdown('<h1 class="main-header">🤖 Sentiment Analysis ML Platform</h1>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar para status da API
with st.sidebar:
    st.header("🔧 Status do Sistema")

    # Verifica status da API
    health_result = api_client.health_check()

    # Status da API Key
    if api_client.has_api_key():
        st.success("🔑 API Key: Configurada")
    else:
        st.error("🔑 API Key: Não configurada")
        st.warning("Configure a variável de ambiente:\n`API_KEY` ou `DEEPLEARNING_API_KEY`")

    # Status da API
    if health_result["success"]:
        st.success("✅ Deep Learning API: Online")
        if "data" in health_result:
            api_data = health_result["data"]
            st.caption(f"App: {api_data.get('app', 'N/A')} | Versão: {api_data.get('version', 'N/A')}")
    else:
        status = health_result.get("status", "unknown")
        message = health_result.get("message", "Erro desconhecido")

        if status == "api_key_missing":
            st.error("❌ API Key não configurada")
        elif status == "unauthorized":
            st.error("❌ API Key inválida")
        elif status == "timeout":
            st.error("❌ Timeout na API")
        elif status == "connection_error":
            st.error("❌ Erro de conexão")
        else:
            st.error("❌ Deep Learning API: Offline")

        st.warning(f"Erro: {message}")

        if status in ["connection_error", "timeout"]:
            st.info("Verifique se a API está acessível em:\nhttps://deeplearning-infnet-project-api-966997855008.europe-west1.run.app")

    st.markdown("---")
    st.header("📋 Instruções")
    st.markdown("""
    **Para usar esta plataforma:**

    1. **Baseline Analysis**: Analise sentimentos com modelo baseline
    2. **Production Analysis**: Analise com modelo de produção otimizado
    3. **Model Testing**: Teste e compare modelos
    4. **API Status**: Monitore o status dos serviços
    """)

# Abas principais
tab1, tab2, tab3, tab4, tab5 = st.tabs(["🤖 Baseline Analysis", "🚀 Production Analysis", "🧪 Model Testing", "🏗️ Architecture", "📋 API Status"])

with tab1:
    render_baseline_training_tab(api_client, health_result["success"])

with tab2:
    render_production_analysis_tab(api_client, health_result["success"])

with tab3:
    render_model_testing_tab(api_client, health_result["success"])

with tab4:
    render_architecture_diagram_tab(api_client, health_result["success"])

with tab5:
    render_api_status_tab(api_client, health_result["success"])

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>"
    "🤖 Sentiment Analysis ML Platform | Powered by Streamlit, FastAPI & MLflow"
    "</div>",
    unsafe_allow_html=True
)