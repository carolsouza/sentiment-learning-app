import streamlit as st
import requests
from api_client import DeepLearningAPIClient


def render_api_status_tab(api_client, api_status):
    """Renderiza a tab de API Status"""
    st.header("📋 Status da API")

    col1, col2 = st.columns(2)

    with col1:
        _render_services_status(api_status)

    with col2:
        _render_useful_links(api_status)


def _render_services_status(api_status):
    """Renderiza o status dos serviços"""
    st.subheader("🔧 Status dos Serviços")

    # Status da Deep Learning API
    if api_status:
        st.success("✅ Deep Learning API: Online")
    else:
        st.error("❌ Deep Learning API: Offline")

    # Status do MLflow via API
    mlflow_result = _check_mlflow_status_via_api()

    if mlflow_result["success"]:
        st.success("✅ MLflow Server: Online")
    else:
        status = mlflow_result.get("status", "unknown")
        message = mlflow_result.get("message", "Erro desconhecido")

        if status == "api_key_missing":
            st.error("❌ MLflow: API Key não configurada")
        elif status == "unauthorized":
            st.error("❌ MLflow: API Key inválida")
        elif status == "timeout":
            st.error("❌ MLflow: Timeout")
        elif status == "connection_error":
            st.error("❌ MLflow: Erro de conexão")
        else:
            st.error("❌ MLflow Server: Offline")

        st.warning(f"Erro: {message}")

    # Botão de refresh
    if st.button("🔄 Atualizar Status"):
        st.rerun()


def _render_useful_links(api_status):
    """Renderiza links úteis"""
    st.subheader("🔗 Links Úteis")

    st.markdown("**Deep Learning API:**")
    if api_status:
        # Pega a URL base da API
        api_client = DeepLearningAPIClient()
        api_url = api_client.api_base_url

        st.markdown(f"• [📖 API Documentation]({api_url}/docs)")
    else:
        st.write("• API Docs: Indisponível (API offline)")

    st.markdown("**MLflow Server:**")
    mlflow_result = _check_mlflow_status_via_api()
    if mlflow_result["success"] and mlflow_result.get("mlflow_uri"):
        mlflow_uri = mlflow_result["mlflow_uri"]
        st.markdown(f"• [🔬 MLflow Dashboard]({mlflow_uri})")
    else:
        st.write("• MLflow Dashboard: Indisponível")


def _check_mlflow_status_via_api():
    """Verifica o status do MLflow via API"""
    api_client = DeepLearningAPIClient()
    return api_client.check_mlflow_health()


