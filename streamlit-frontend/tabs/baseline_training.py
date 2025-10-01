import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from typing import List, Dict, Any


@st.cache_data(ttl=300, show_spinner=False)
def _fetch_baseline_metrics(_api_client):
    """Busca métricas do baseline com cache de 5 minutos"""
    return _api_client.get_model_details("Embedding Baseline")


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
    col1, col2 = st.columns([3, 1])
    with col1:
        st.subheader("📊 Métricas do Modelo Baseline")
    with col2:
        if st.button("🔄 Atualizar", key="refresh_baseline"):
            _fetch_baseline_metrics.clear()
            st.rerun()

    with st.spinner("🔄 Carregando métricas..."):
        result = _fetch_baseline_metrics(api_client)

    if not result["success"]:
        st.error(f"❌ {result.get('error', 'Erro ao carregar métricas')}")
        return

    data = result["data"]

    if not data.get("found", False):
        st.warning("🔍 Modelo 'Embedding Baseline' não encontrado no MLflow")
        return

    # Info do modelo
    model_info = data["model_info"]
    st.success(f"✅ Modelo: **{model_info['model_name']}** (v{model_info['version']}) - Stage: {model_info['stage']}")

    # Métricas principais - Test
    st.markdown("### 🎯 Métricas de Teste")
    metrics = data["metrics"]

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("F1-Score", f"{metrics.get('test_f1', 0):.4f}")
    with col2:
        st.metric("Accuracy", f"{metrics.get('test_accuracy', 0):.4f}")
    with col3:
        st.metric("ROC-AUC", f"{metrics.get('test_roc_auc', 0):.4f}")
    with col4:
        st.metric("Threshold", f"{metrics.get('best_threshold', 0.5):.3f}")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Precision", f"{metrics.get('test_precision', 0):.4f}")
    with col2:
        st.metric("Recall", f"{metrics.get('test_recall', 0):.4f}")
    with col3:
        if metrics.get('best_epoch'):
            st.metric("Best Epoch", f"{int(metrics.get('best_epoch', 0))}")
    with col4:
        if metrics.get('best_f1_val'):
            st.metric("Best F1 (Val)", f"{metrics.get('best_f1_val', 0):.4f}")

    # Dataset Info
    dataset = data.get("dataset_info", {})
    if dataset:
        st.markdown("### 📦 Dataset")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Train", f"{int(dataset.get('train_size', 0)):,}")
        with col2:
            st.metric("Validation", f"{int(dataset.get('val_size', 0)):,}")
        with col3:
            st.metric("Test", f"{int(dataset.get('test_size', 0)):,}")

    # Parâmetros
    st.markdown("---")
    st.markdown("### 🔧 Hiperparâmetros")
    params = data.get("params", {})

    params_col1, params_col2, params_col3 = st.columns(3)

    with params_col1:
        st.write("**Arquitetura:**")
        for key in ['model_type', 'vocab_size', 'embed_dim']:
            if key in params:
                st.write(f"• {key}: {params[key]}")

    with params_col2:
        st.write("**Treinamento:**")
        for key in ['optimizer', 'learning_rate', 'batch_size']:
            if key in params:
                st.write(f"• {key}: {params[key]}")

    with params_col3:
        st.write("**Outros:**")
        for key in ['dropout_rate', 'total_params']:
            if key in params:
                st.write(f"• {key}: {params[key]}")

    # Gráficos
    st.markdown("---")
    st.markdown("### 📊 Visualizações")

    training_history = data.get("training_history", {})
    cm = data.get("confusion_matrix", {})

    # Verificar se há dados para mostrar
    has_loss = training_history.get("loss") and training_history.get("val_loss")
    has_auc = training_history.get("auc") and training_history.get("val_auc")
    has_accuracy = training_history.get("accuracy") and training_history.get("val_accuracy")
    has_precision = training_history.get("precision") and training_history.get("val_precision")
    has_recall = training_history.get("recall") and training_history.get("val_recall")
    has_confusion_matrix = cm.get('tn') is not None or cm.get('tp') is not None

    if has_loss or has_auc or has_accuracy or has_precision or has_recall or has_confusion_matrix:
        # Criar subplot com 3 linhas x 2 colunas
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=("Loss", "AUC", "Accuracy", "Precision", "Recall", "Confusion Matrix"),
            specs=[
                [{"type": "scatter"}, {"type": "scatter"}],
                [{"type": "scatter"}, {"type": "scatter"}],
                [{"type": "scatter"}, {"type": "heatmap"}]
            ]
        )

        # Gráfico de Loss
        if has_loss:
            loss_data = training_history["loss"]
            val_loss_data = training_history["val_loss"]

            epochs = [item["epoch"] for item in loss_data]
            values = [item["value"] for item in loss_data]
            fig.add_trace(go.Scatter(x=epochs, y=values,
                                   mode='lines', name='Train Loss',
                                   line=dict(color='blue'), showlegend=True), row=1, col=1)

            epochs = [item["epoch"] for item in val_loss_data]
            values = [item["value"] for item in val_loss_data]
            fig.add_trace(go.Scatter(x=epochs, y=values,
                                   mode='lines', name='Val Loss',
                                   line=dict(color='red'), showlegend=True), row=1, col=1)

        # Gráfico de AUC
        if has_auc:
            auc_data = training_history["auc"]
            val_auc_data = training_history["val_auc"]

            epochs = [item["epoch"] for item in auc_data]
            values = [item["value"] for item in auc_data]
            fig.add_trace(go.Scatter(x=epochs, y=values,
                                   mode='lines', name='Train AUC',
                                   line=dict(color='green'), showlegend=False), row=1, col=2)

            epochs = [item["epoch"] for item in val_auc_data]
            values = [item["value"] for item in val_auc_data]
            fig.add_trace(go.Scatter(x=epochs, y=values,
                                   mode='lines', name='Val AUC',
                                   line=dict(color='orange'), showlegend=False), row=1, col=2)

        # Gráfico de Accuracy
        if has_accuracy:
            acc_data = training_history["accuracy"]
            val_acc_data = training_history["val_accuracy"]

            epochs = [item["epoch"] for item in acc_data]
            values = [item["value"] for item in acc_data]
            fig.add_trace(go.Scatter(x=epochs, y=values,
                                   mode='lines', name='Train Acc',
                                   line=dict(color='purple'), showlegend=False), row=2, col=1)

            epochs = [item["epoch"] for item in val_acc_data]
            values = [item["value"] for item in val_acc_data]
            fig.add_trace(go.Scatter(x=epochs, y=values,
                                   mode='lines', name='Val Acc',
                                   line=dict(color='brown'), showlegend=False), row=2, col=1)

        # Gráfico de Precision
        if has_precision:
            prec_data = training_history["precision"]
            val_prec_data = training_history["val_precision"]

            epochs = [item["epoch"] for item in prec_data]
            values = [item["value"] for item in prec_data]
            fig.add_trace(go.Scatter(x=epochs, y=values,
                                   mode='lines', name='Train Precision',
                                   line=dict(color='cyan'), showlegend=False), row=2, col=2)

            epochs = [item["epoch"] for item in val_prec_data]
            values = [item["value"] for item in val_prec_data]
            fig.add_trace(go.Scatter(x=epochs, y=values,
                                   mode='lines', name='Val Precision',
                                   line=dict(color='magenta'), showlegend=False), row=2, col=2)

        # Gráfico de Recall
        if has_recall:
            rec_data = training_history["recall"]
            val_rec_data = training_history["val_recall"]

            epochs = [item["epoch"] for item in rec_data]
            values = [item["value"] for item in rec_data]
            fig.add_trace(go.Scatter(x=epochs, y=values,
                                   mode='lines', name='Train Recall',
                                   line=dict(color='yellow'), showlegend=False), row=3, col=1)

            epochs = [item["epoch"] for item in val_rec_data]
            values = [item["value"] for item in val_rec_data]
            fig.add_trace(go.Scatter(x=epochs, y=values,
                                   mode='lines', name='Val Recall',
                                   line=dict(color='pink'), showlegend=False), row=3, col=1)

        # Confusion Matrix
        if has_confusion_matrix:
            confusion_matrix = [
                [cm.get('tn', 0), cm.get('fp', 0)],
                [cm.get('fn', 0), cm.get('tp', 0)]
            ]

            fig.add_trace(go.Heatmap(
                z=confusion_matrix,
                x=['Negative', 'Positive'],
                y=['Negative', 'Positive'],
                text=confusion_matrix,
                texttemplate='%{text}',
                textfont={"size": 16},
                colorscale='Blues',
                showscale=False
            ), row=3, col=2)

        fig.update_xaxes(title_text="Época", row=1, col=1)
        fig.update_xaxes(title_text="Época", row=1, col=2)
        fig.update_xaxes(title_text="Época", row=2, col=1)
        fig.update_xaxes(title_text="Época", row=2, col=2)
        fig.update_xaxes(title_text="Época", row=3, col=1)
        fig.update_xaxes(title_text="Predicted", row=3, col=2)
        fig.update_yaxes(title_text="Actual", row=3, col=2)

        fig.update_layout(
            height=900,
            showlegend=True,
            hovermode='x unified'
        )

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("📊 Dados de visualização não disponíveis")

    # Link MLflow
    st.markdown("---")
    st.info(f"🔗 [Ver no MLflow]({model_info['mlflow_uri']})")


