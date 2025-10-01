import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from typing import List, Dict, Any


@st.cache_data(ttl=300, show_spinner=False)
def _fetch_production_metrics(_api_client):
    """Busca métricas do modelo de produção com cache de 5 minutos"""
    return _api_client.get_model_details("BiLSTM - Deep Learning")


def render_production_analysis_tab(api_client, api_status):
    """Renderiza a tab de Production Analysis - métricas e análise do modelo"""
    st.header("🚀 Modelo de Produção - Métricas e Análise")

    # Apenas métricas do modelo
    _render_api_metrics_production(api_client)


def _render_production_analysis_interface(api_client):
    """Interface simples para análise de sentimento com modelo de produção"""

    # Informação sobre o modelo
    st.info("🚀 **Modelo de Produção**: BiLSTM otimizado com dados balanceados para melhor performance.")

    # Seção de teste individual
    st.subheader("📝 Teste de Texto")

    user_text = st.text_area(
        "Digite um texto para analisar:",
        placeholder="Ex: O atendimento foi excepcional e o produto chegou rapidamente!",
        height=100
    )

    if st.button("🚀 Analisar com Modelo de Produção", use_container_width=True):
        if user_text.strip():
            with st.spinner("Analisando com modelo de produção..."):
                result = api_client.predict_production(user_text.strip())

            if result["success"]:
                prediction = result["prediction"]
                _display_production_result(prediction)
            else:
                st.error(f"❌ Erro: {result.get('error', 'Erro desconhecido')}")
        else:
            st.warning("⚠️ Digite um texto para análise")

    # Exemplos rápidos
    st.markdown("---")
    st.subheader("💡 Exemplos Rápidos")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("😀 Exemplo Positivo"):
            _test_production_example(api_client, "O produto superou todas as minhas expectativas! Qualidade excepcional e entrega rápida.")

    with col2:
        if st.button("😞 Exemplo Negativo"):
            _test_production_example(api_client, "Produto de péssima qualidade, não funcionou e o suporte foi terrível.")


def _render_models_comparison(api_client):
    """Interface para comparação entre modelos"""
    st.subheader("⚖️ Comparação: Baseline vs Produção")

    comparison_text = st.text_input(
        "Digite um texto para comparar os dois modelos:",
        placeholder="Ex: O produto é bom mas o preço está alto"
    )

    if st.button("🔍 Comparar Modelos", use_container_width=True):
        if comparison_text.strip():
            _compare_models(api_client, comparison_text.strip())
        else:
            st.warning("⚠️ Digite um texto para comparação")



def _display_production_result(prediction):
    """Exibe resultado simples da predição do modelo de produção"""
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

    # Barra de progresso visual com indicação de produção
    progress_color = "green" if sentiment == "positivo" else "red"
    st.markdown(f"""
    <div style="background-color: #f0f0f0; border-radius: 10px; padding: 10px; margin: 10px 0;">
        <div style="background-color: {progress_color}; height: 20px; width: {score*100}%; border-radius: 10px; text-align: center; color: white; font-weight: bold;">
            🚀 Produção: {score:.3f}
        </div>
    </div>
    """, unsafe_allow_html=True)


def _test_production_example(api_client, text):
    """Testa um exemplo específico com modelo de produção"""
    with st.spinner("Analisando exemplo com modelo de produção..."):
        result = api_client.predict_production(text)

    if result["success"]:
        st.write(f"**Texto:** {text}")
        _display_production_result(result["prediction"])
    else:
        st.error(f"❌ Erro: {result.get('error', 'Erro desconhecido')}")


def _compare_models(api_client, text):
    """Compara os resultados do modelo baseline e produção"""
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🤖 Modelo Baseline")
        with st.spinner("Analisando com baseline..."):
            baseline_result = api_client.predict_baseline(text)

        if baseline_result["success"]:
            baseline_pred = baseline_result["prediction"]
            st.metric("Sentimento", baseline_pred["sentiment"].title())
            st.metric("Score", f"{baseline_pred['score']:.3f}")
            st.metric("Confiança", f"{baseline_pred['confidence']:.1%}")
        else:
            st.error("❌ Erro no modelo baseline")

    with col2:
        st.subheader("🚀 Modelo Produção")
        with st.spinner("Analisando com produção..."):
            prod_result = api_client.predict_production(text)

        if prod_result["success"]:
            prod_pred = prod_result["prediction"]
            st.metric("Sentimento", prod_pred["sentiment"].title())
            st.metric("Score", f"{prod_pred['score']:.3f}")
            st.metric("Confiança", f"{prod_pred['confidence']:.1%}")
        else:
            st.error("❌ Erro no modelo de produção")

    # Comparação visual se ambos funcionaram
    if baseline_result.get("success") and prod_result.get("success"):
        st.markdown("---")
        st.subheader("📊 Comparação Visual")

        baseline_score = baseline_pred["score"]
        prod_score = prod_pred["score"]

        # Gráfico de barras comparativo
        fig = go.Figure(data=[
            go.Bar(name='Baseline', x=['Score'], y=[baseline_score], marker_color='lightblue'),
            go.Bar(name='Produção', x=['Score'], y=[prod_score], marker_color='darkgreen')
        ])

        fig.update_layout(
            title="Comparação de Scores",
            yaxis_title="Score",
            yaxis=dict(range=[0, 1]),
            barmode='group',
            height=400
        )

        st.plotly_chart(fig, use_container_width=True)

        # Diferença
        diff = abs(prod_score - baseline_score)
        if diff > 0.1:
            if prod_score > baseline_score:
                st.success(f"🚀 Modelo de produção mais positivo (+{diff:.3f})")
            else:
                st.info(f"🤖 Modelo baseline mais positivo (+{diff:.3f})")
        else:
            st.info(f"⚖️ Modelos concordam (diferença: {diff:.3f})")


def _render_api_metrics_production(api_client):
    """Renderiza métricas do modelo de produção via API"""
    col1, col2 = st.columns([3, 1])
    with col1:
        st.subheader("📊 Métricas do Modelo de Produção")
    with col2:
        if st.button("🔄 Atualizar", key="refresh_production"):
            _fetch_production_metrics.clear()
            st.rerun()

    with st.spinner("🔄 Carregando métricas..."):
        result = _fetch_production_metrics(api_client)

    if not result["success"]:
        st.error(f"❌ {result.get('error', 'Erro ao carregar métricas')}")
        return

    data = result["data"]

    if not data.get("found", False):
        st.warning("🔍 Modelo 'BiLSTM - Deep Learning' não encontrado no MLflow")
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
        for key in ['model_type', 'lstm_units', 'vocab_size', 'embed_dim']:
            if key in params:
                st.write(f"• {key}: {params[key]}")

    with params_col2:
        st.write("**Treinamento:**")
        for key in ['optimizer', 'learning_rate', 'batch_size', 'clipnorm']:
            if key in params:
                st.write(f"• {key}: {params[key]}")

    with params_col3:
        st.write("**Outros:**")
        for key in ['dropout_rate', 'recurrent_dropout', 'total_params']:
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


def _render_api_models_comparison(api_client):
    """Renderiza comparação entre modelos via API"""
    st.subheader("⚖️ Comparação: Baseline vs Produção")

    with st.spinner("🔄 Carregando comparação entre modelos..."):
        result = api_client.get_model_comparison()

    if not result["success"]:
        st.error(f"❌ Erro ao carregar comparação: {result.get('error', 'Erro desconhecido')}")
        return

    data = result["data"]

    if not data.get("comparison_available", False):
        st.warning("🔍 Comparação não disponível")
        st.info(f"Baseline encontrado: {'✅' if data.get('baseline_found', False) else '❌'}")
        st.info(f"Produção encontrado: {'✅' if data.get('production_found', False) else '❌'}")
        return

    # Resumo da comparação
    summary = data["summary"]
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("🏆 Vitórias Produção", summary["production_wins"])

    with col2:
        st.metric("🥈 Vitórias Baseline", summary["baseline_wins"])

    with col3:
        st.metric("🤝 Empates", summary["ties"])

    # Tabela de comparação
    st.markdown("---")
    st.subheader("📊 Métricas Detalhadas")

    comparison_data = data["comparison"]

    # Prepara dados para a tabela
    table_data = []
    for item in comparison_data:
        table_data.append({
            "Métrica": item["metric"].replace("_", " ").title(),
            "Baseline": f"{item['baseline']:.4f}",
            "Produção": f"{item['production']:.4f}",
            "Diferença": f"{item['difference']:+.4f}",
            "Melhor": item["better_model"].title()
        })

    df_comparison = pd.DataFrame(table_data)
    st.dataframe(df_comparison, use_container_width=True, hide_index=True)

    # Gráfico comparativo
    st.markdown("---")
    st.subheader("📈 Visualização Comparativa")

    # Seleciona métricas principais para o gráfico
    metrics_for_chart = ["test_f1", "test_roc_auc"]
    baseline_values = []
    production_values = []
    metric_names = []

    for item in comparison_data:
        if item["metric"] in metrics_for_chart:
            baseline_values.append(item["baseline"])
            production_values.append(item["production"])
            metric_names.append(item["metric"].replace("_", " ").title())

    if baseline_values and production_values:
        fig = go.Figure(data=[
            go.Bar(name='Baseline (DNN)', x=metric_names, y=baseline_values, marker_color='lightblue'),
            go.Bar(name='Produção (BiLSTM)', x=metric_names, y=production_values, marker_color='darkgreen')
        ])

        fig.update_layout(
            title="Comparação de Performance: Baseline vs Produção",
            yaxis_title="Score",
            yaxis=dict(range=[0, 1]),
            barmode='group',
            height=400
        )

        st.plotly_chart(fig, use_container_width=True)

    # Conclusão
    st.markdown("---")
    if summary["production_wins"] > summary["baseline_wins"]:
        st.success("🚀 **Modelo de Produção (BiLSTM)** apresenta melhor performance geral!")
    elif summary["baseline_wins"] > summary["production_wins"]:
        st.info("🤖 **Modelo Baseline (DNN)** apresenta melhor performance geral!")
    else:
        st.warning("⚖️ **Modelos empatados** - Performance similar entre os dois modelos!")


def _render_old_mlflow_metrics_production():
    """Renderiza métricas do modelo de produção do MLflow"""
    st.subheader("📊 Métricas do Modelo de Produção (BiLSTM)")

    try:
        # Busca experimentos do modelo de produção
        client = mlflow.tracking.MlflowClient()

        # Busca runs com o nome "production_bilstm64_dnn"
        runs = client.search_runs(
            experiment_ids=["0"],  # Default experiment
            filter_string="run_name = 'production_bilstm64_dnn'",
            order_by=["start_time DESC"],
            max_results=5
        )

        if not runs:
            st.warning("🔍 Nenhum experimento de produção encontrado no MLflow")
            st.info("Execute o script de treino de produção primeiro:\n```\ncd models-training\npython train_production.py\n```")
            return

        # Pega o run mais recente
        latest_run = runs[0]
        run_id = latest_run.info.run_id

        st.success(f"✅ Experimento encontrado: {run_id[:8]}...")

        # Métricas principais
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            test_f1 = latest_run.data.metrics.get("test_f1", 0)
            st.metric("🎯 F1-Score (Test)", f"{test_f1:.4f}")

        with col2:
            test_roc_auc = latest_run.data.metrics.get("test_roc_auc", 0)
            st.metric("📈 ROC-AUC (Test)", f"{test_roc_auc:.4f}")

        with col3:
            val_loss = latest_run.data.metrics.get("val_loss", 0)
            st.metric("📉 Validation Loss", f"{val_loss:.4f}")

        with col4:
            best_threshold = latest_run.data.metrics.get("best_threshold", 0.5)
            st.metric("⚖️ Best Threshold", f"{best_threshold:.3f}")

        # Parâmetros do modelo
        st.markdown("---")
        st.subheader("🔧 Parâmetros do Modelo")

        params_col1, params_col2, params_col3 = st.columns(3)

        with params_col1:
            st.write("**Arquitetura:**")
            st.write(f"• Tipo: {latest_run.data.params.get('model_type', 'N/A')}")
            st.write(f"• LSTM Units: {latest_run.data.params.get('lstm_units', 'N/A')}")
            st.write(f"• Vocab Size: {latest_run.data.params.get('vocab_size', 'N/A')}")
            st.write(f"• Embed Dim: {latest_run.data.params.get('embed_dim', 'N/A')}")

        with params_col2:
            st.write("**Treinamento:**")
            st.write(f"• Optimizer: {latest_run.data.params.get('optimizer', 'N/A')}")
            st.write(f"• Learning Rate: {latest_run.data.params.get('learning_rate', 'N/A')}")
            st.write(f"• Batch Size: {latest_run.data.params.get('batch_size', 'N/A')}")
            st.write(f"• Clipnorm: {latest_run.data.params.get('clipnorm', 'N/A')}")

        with params_col3:
            st.write("**Regularização:**")
            st.write(f"• Dropout: {latest_run.data.params.get('dropout_rate', 'N/A')}")
            st.write(f"• Recurrent Dropout: {latest_run.data.params.get('recurrent_dropout', 'N/A')}")
            st.write(f"• Total Params: {latest_run.data.params.get('total_params', 'N/A')}")

        # Histórico de métricas (se disponível)
        st.markdown("---")
        st.subheader("📈 Histórico de Treinamento")

        try:
            # Busca histórico de métricas
            metric_history = client.get_metric_history(run_id, "loss")
            val_metric_history = client.get_metric_history(run_id, "val_loss")

            if metric_history and val_metric_history:
                # Cria DataFrame para plotar
                train_data = pd.DataFrame([(m.step, m.value) for m in metric_history], columns=['epoch', 'train_loss'])
                val_data = pd.DataFrame([(m.step, m.value) for m in val_metric_history], columns=['epoch', 'val_loss'])

                # Gráfico de loss
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=train_data['epoch'], y=train_data['train_loss'],
                                       mode='lines', name='Training Loss', line=dict(color='blue')))
                fig.add_trace(go.Scatter(x=val_data['epoch'], y=val_data['val_loss'],
                                       mode='lines', name='Validation Loss', line=dict(color='red')))

                fig.update_layout(
                    title="Evolução do Loss durante o Treinamento (BiLSTM)",
                    xaxis_title="Época",
                    yaxis_title="Loss",
                    hovermode='x unified'
                )

                st.plotly_chart(fig, use_container_width=True)

                # Busca outras métricas se disponíveis
                try:
                    auc_history = client.get_metric_history(run_id, "val_auc")
                    val_auc_history = client.get_metric_history(run_id, "val_auc")

                    if auc_history:
                        auc_data = pd.DataFrame([(m.step, m.value) for m in auc_history], columns=['epoch', 'val_auc'])

                        # Gráfico de AUC
                        fig_auc = go.Figure()
                        fig_auc.add_trace(go.Scatter(x=auc_data['epoch'], y=auc_data['val_auc'],
                                                   mode='lines', name='Validation AUC', line=dict(color='green')))

                        fig_auc.update_layout(
                            title="Evolução do AUC durante o Treinamento",
                            xaxis_title="Época",
                            yaxis_title="AUC",
                            hovermode='x unified'
                        )

                        st.plotly_chart(fig_auc, use_container_width=True)
                except:
                    pass  # AUC pode não estar disponível

            else:
                st.info("📊 Histórico de métricas não disponível para este run")

        except Exception as e:
            st.warning(f"⚠️ Não foi possível carregar o histórico: {str(e)}")

        # Comparação com baseline (se disponível)
        st.markdown("---")
        st.subheader("⚖️ Comparação com Baseline")

        try:
            # Busca também o baseline para comparar
            baseline_runs = client.search_runs(
                experiment_ids=["0"],
                filter_string="run_name = 'baseline_dnn_pool'",
                order_by=["start_time DESC"],
                max_results=1
            )

            if baseline_runs:
                baseline_run = baseline_runs[0]

                # Tabela comparativa
                comparison_data = {
                    "Métrica": ["F1-Score (Test)", "ROC-AUC (Test)", "Validation Loss", "Best Threshold"],
                    "Baseline (DNN)": [
                        f"{baseline_run.data.metrics.get('test_f1', 0):.4f}",
                        f"{baseline_run.data.metrics.get('test_roc_auc', 0):.4f}",
                        f"{baseline_run.data.metrics.get('val_loss', 0):.4f}",
                        f"{baseline_run.data.metrics.get('best_threshold', 0.5):.3f}"
                    ],
                    "Produção (BiLSTM)": [
                        f"{latest_run.data.metrics.get('test_f1', 0):.4f}",
                        f"{latest_run.data.metrics.get('test_roc_auc', 0):.4f}",
                        f"{latest_run.data.metrics.get('val_loss', 0):.4f}",
                        f"{latest_run.data.metrics.get('best_threshold', 0.5):.3f}"
                    ]
                }

                df_comparison = pd.DataFrame(comparison_data)
                st.dataframe(df_comparison, use_container_width=True, hide_index=True)

                # Gráfico comparativo
                fig_comp = go.Figure(data=[
                    go.Bar(name='Baseline (DNN)', x=['F1-Score', 'ROC-AUC'],
                          y=[baseline_run.data.metrics.get('test_f1', 0),
                             baseline_run.data.metrics.get('test_roc_auc', 0)],
                          marker_color='lightblue'),
                    go.Bar(name='Produção (BiLSTM)', x=['F1-Score', 'ROC-AUC'],
                          y=[latest_run.data.metrics.get('test_f1', 0),
                             latest_run.data.metrics.get('test_roc_auc', 0)],
                          marker_color='darkgreen')
                ])

                fig_comp.update_layout(
                    title="Comparação de Performance: Baseline vs Produção",
                    yaxis_title="Score",
                    yaxis=dict(range=[0, 1]),
                    barmode='group',
                    height=400
                )

                st.plotly_chart(fig_comp, use_container_width=True)

            else:
                st.info("🔍 Baseline não encontrado para comparação")

        except Exception as e:
            st.warning(f"⚠️ Erro na comparação: {str(e)}")

        # Informações do experimento
        st.markdown("---")
        st.subheader("ℹ️ Informações do Experimento")

        info_col1, info_col2 = st.columns(2)

        with info_col1:
            st.write(f"**Run ID:** `{run_id}`")
            st.write(f"**Status:** {latest_run.info.status}")
            st.write(f"**Início:** {pd.to_datetime(latest_run.info.start_time, unit='ms').strftime('%Y-%m-%d %H:%M:%S')}")

        with info_col2:
            if latest_run.info.end_time:
                end_time = pd.to_datetime(latest_run.info.end_time, unit='ms')
                start_time = pd.to_datetime(latest_run.info.start_time, unit='ms')
                duration = end_time - start_time
                st.write(f"**Fim:** {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
                st.write(f"**Duração:** {duration}")

        # Link para MLflow UI
        st.markdown("---")
        st.info(f"🔗 [Ver experimento completo no MLflow](https://mlflow-server-273169854208.us-central1.run.app/#/experiments/0/runs/{run_id})")

    except Exception as e:
        st.error(f"❌ Erro ao conectar com MLflow: {str(e)}")
        st.info("Verifique se o MLflow server está acessível em: https://mlflow-server-273169854208.us-central1.run.app")


