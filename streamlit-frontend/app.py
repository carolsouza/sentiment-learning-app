import streamlit as st
from api_client import MLAPIClient
from tabs.dataset_training import render_dataset_training_tab
from tabs.baseline_training import render_baseline_training_tab
from tabs.model_testing import render_model_testing_tab
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

# Inicializa cliente da API
# Temporariamente sem cache para garantir que a nova versão seja carregada
api_client = MLAPIClient()

# Debug: Verifica se o método existe
if not hasattr(api_client, 'upload_dataset'):
    st.error("⚠️ Método upload_dataset não encontrado. Recarregue a página.")

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
    api_status = api_client.health_check()
    if api_status:
        st.success("✅ ML API: Online")
    else:
        st.error("❌ ML API: Offline")
        st.warning("Verifique se a API está rodando em http://localhost:8000")

    st.markdown("---")
    st.header("📋 Instruções")
    st.markdown("""
    **Para usar esta plataforma:**

    1. **Dataset Upload**: Faça upload do arquivo CSV Amazon Fine Foods
    2. **Baseline Training**: Configure e treine o modelo Naive Bayes
    3. **Model Testing**: Teste o modelo com textos customizados
    4. **API Status**: Monitore o status dos serviços
    """)

# Abas principais
tab1, tab2, tab3, tab4 = st.tabs(["📊 Dataset Upload", "🚀 Baseline Training", "🧪 Model Testing", "📋 API Status"])

with tab1:
    render_dataset_training_tab(api_client, api_status)

with tab2:
    render_baseline_training_tab(api_client, api_status)

with tab3:
    render_model_testing_tab(api_client, api_status)

with tab4:
    render_api_status_tab(api_client, api_status)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>"
    "🤖 Sentiment Analysis ML Platform | Powered by Streamlit, FastAPI & MLflow"
    "</div>",
    unsafe_allow_html=True
)