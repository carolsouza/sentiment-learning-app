import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from typing import List, Dict, Any


def render_baseline_training_tab(api_client, api_status):
    """Renderiza a tab de Baseline - análise e métricas do modelo"""
    st.header("🤖 Modelo Baseline - Métricas e Análise")

    # Apenas métricas do modelo
    _render_api_metrics_baseline(api_client)


def _render_simple_analysis_interface(api_client):
    """Interface simples para análise de sentimento"""

    # Seção de teste individual
    st.subheader("📝 Teste de Texto")

    user_text = st.text_area(
        "Digite um texto para analisar:",
        placeholder="Ex: Este produto é excelente! Recomendo para todos.",
        height=100
    )

    if st.button("🚀 Analisar Sentimento", use_container_width=True):
        if user_text.strip():
            with st.spinner("Analisando..."):
                result = api_client.predict_baseline(user_text.strip())

            if result["success"]:
                prediction = result["prediction"]
                _display_simple_result(prediction)
            else:
                st.error(f"❌ Erro: {result.get('error', 'Erro desconhecido')}")
        else:
            st.warning("⚠️ Digite um texto para análise")

    # Exemplos rápidos
    st.markdown("---")
    st.subheader("💡 Exemplos Rápidos")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("😀 Texto Positivo"):
            _test_example(api_client, "Este produto é fantástico! Superou minhas expectativas.")

    with col2:
        if st.button("😞 Texto Negativo"):
            _test_example(api_client, "Produto terrível, não funcionou e chegou quebrado.")



def _display_simple_result(prediction):
    """Exibe resultado simples da predição"""
    sentiment = prediction["sentiment"]
    score = prediction["score"]
    confidence = prediction["confidence"]

    # Resultado principal
    col1, col2, col3 = st.columns(3)

    with col1:
        emoji = "😀" if sentiment == "positivo" else "😞"
        st.metric(f"{emoji} Sentimento", sentiment.title())

    with col2:
        st.metric("📊 Score", f"{score:.3f}")

    with col3:
        st.metric("🎯 Confiança", f"{confidence:.1%}")

    # Barra de progresso visual
    progress_color = "green" if sentiment == "positivo" else "red"
    st.markdown(f"""
    <div style="background-color: #f0f0f0; border-radius: 10px; padding: 10px; margin: 10px 0;">
        <div style="background-color: {progress_color}; height: 20px; width: {score*100}%; border-radius: 10px; text-align: center; color: white; font-weight: bold;">
            {score:.3f}
        </div>
    </div>
    """, unsafe_allow_html=True)


def _test_example(api_client, text):
    """Testa um exemplo específico"""
    with st.spinner("Analisando exemplo..."):
        result = api_client.predict_baseline(text)

    if result["success"]:
        st.write(f"**Texto:** {text}")
        _display_simple_result(result["prediction"])
    else:
        st.error(f"❌ Erro: {result.get('error', 'Erro desconhecido')}")


def _render_api_metrics_baseline(api_client):
    """Renderiza métricas do modelo baseline via API"""
    st.subheader("📊 Métricas do Modelo Baseline (DNN Pool)")

    with st.spinner("🔄 Carregando métricas do modelo baseline..."):
        result = api_client.get_baseline_metrics()

    if not result["success"]:
        st.error(f"❌ Erro ao carregar métricas: {result.get('error', 'Erro desconhecido')}")
        return

    data = result["data"]

    if not data.get("found", False):
        st.warning("🔍 Nenhum experimento de baseline encontrado no MLflow")
        return

    st.success(f"✅ Experimento encontrado: {data['run_id']}...")

    # Métricas principais
    metrics = data["metrics"]
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("🎯 F1-Score (Test)", f"{metrics.get('test_f1', 0):.4f}")

    with col2:
        st.metric("📈 ROC-AUC (Test)", f"{metrics.get('test_roc_auc', 0):.4f}")

    with col3:
        st.metric("📉 Validation Loss", f"{metrics.get('val_loss', 0):.4f}")

    with col4:
        st.metric("⚖️ Best Threshold", f"{metrics.get('best_threshold', 0.5):.3f}")

    # Parâmetros do modelo
    st.markdown("---")
    st.subheader("🔧 Parâmetros do Modelo")

    params = data["params"]
    params_col1, params_col2, params_col3 = st.columns(3)

    with params_col1:
        st.write("**Arquitetura:**")
        st.write(f"• Tipo: {params.get('model_type', 'N/A')}")
        st.write(f"• Vocab Size: {params.get('vocab_size', 'N/A')}")
        st.write(f"• Embed Dim: {params.get('embed_dim', 'N/A')}")

    with params_col2:
        st.write("**Treinamento:**")
        st.write(f"• Optimizer: {params.get('optimizer', 'N/A')}")
        st.write(f"• Learning Rate: {params.get('learning_rate', 'N/A')}")
        st.write(f"• Batch Size: {params.get('batch_size', 'N/A')}")

    with params_col3:
        st.write("**Regularização:**")
        st.write(f"• Dropout: {params.get('dropout_rate', 'N/A')}")
        st.write(f"• Total Params: {params.get('total_params', 'N/A')}")

    # Histórico de treinamento
    st.markdown("---")
    st.subheader("📈 Histórico de Treinamento")

    training_history = data.get("training_history", {})

    if training_history.get("loss") and training_history.get("val_loss"):
        # Prepara dados para o gráfico
        loss_data = training_history["loss"]
        val_loss_data = training_history["val_loss"]

        # Cria o gráfico
        fig = go.Figure()

        if loss_data:
            epochs = [item["epoch"] for item in loss_data]
            values = [item["value"] for item in loss_data]
            fig.add_trace(go.Scatter(x=epochs, y=values,
                                   mode='lines', name='Training Loss', line=dict(color='blue')))

        if val_loss_data:
            epochs = [item["epoch"] for item in val_loss_data]
            values = [item["value"] for item in val_loss_data]
            fig.add_trace(go.Scatter(x=epochs, y=values,
                                   mode='lines', name='Validation Loss', line=dict(color='red')))

        fig.update_layout(
            title="Evolução do Loss durante o Treinamento",
            xaxis_title="Época",
            yaxis_title="Loss",
            hovermode='x unified'
        )

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("📊 Histórico de métricas não disponível para este run")

    # Informações do experimento
    st.markdown("---")
    st.subheader("ℹ️ Informações do Experimento")

    experiment_info = data["experiment_info"]
    info_col1, info_col2 = st.columns(2)

    with info_col1:
        st.write(f"**Run ID:** `{experiment_info['run_id']}`")
        st.write(f"**Status:** {experiment_info['status']}")
        if experiment_info.get('start_time'):
            start_time = pd.to_datetime(experiment_info['start_time'], unit='ms')
            st.write(f"**Início:** {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    with info_col2:
        if experiment_info.get('end_time'):
            end_time = pd.to_datetime(experiment_info['end_time'], unit='ms')
            start_time = pd.to_datetime(experiment_info['start_time'], unit='ms')
            duration = end_time - start_time
            st.write(f"**Fim:** {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
            st.write(f"**Duração:** {duration}")

    # Link para MLflow UI
    st.markdown("---")
    if experiment_info.get('mlflow_uri'):
        st.info(f"🔗 [Ver experimento completo no MLflow]({experiment_info['mlflow_uri']})")


