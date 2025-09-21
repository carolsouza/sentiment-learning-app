import streamlit as st
import pandas as pd
import plotly.express as px


def render_experiments_tab(api_client, api_status):
    """Renderiza a tab de Experiments"""
    st.header("📈 Histórico de Experimentos")

    if api_status:
        _render_experiments_list(api_client)
    else:
        st.error("❌ API não está disponível para buscar experimentos.")


def _render_experiments_list(api_client):
    """Renderiza a lista de experimentos"""
    # Busca experimentos
    experiments = api_client.get_experiments()

    if experiments:
        st.subheader(f"📊 Total de Experimentos: {len(experiments)}")

        # Converte para DataFrame para melhor visualização
        exp_data = _convert_experiments_to_dataframe(experiments)
        exp_df = pd.DataFrame(exp_data)

        # Tabela de experimentos
        st.subheader("📋 Lista de Experimentos")
        st.dataframe(exp_df, use_container_width=True)

        # Gráficos de métricas ao longo do tempo
        if len(exp_df) > 1:
            _render_metrics_evolution_charts(exp_df)
            _render_best_models_summary(exp_df)

    else:
        st.info("📊 Nenhum experimento encontrado. Execute um treinamento primeiro.")


def _convert_experiments_to_dataframe(experiments):
    """Converte lista de experimentos para formato DataFrame"""
    exp_data = []
    for exp in experiments:
        exp_data.append({
            "Run ID": exp.get("run_id", "")[:8] + "...",
            "Experiment ID": exp.get("experiment_id", ""),
            "Accuracy": exp.get("metrics", {}).get("accuracy", 0),
            "F1-Score": exp.get("metrics", {}).get("f1_score", 0),
            "Precision": exp.get("metrics", {}).get("precision", 0),
            "Recall": exp.get("metrics", {}).get("recall", 0),
            "Status": exp.get("status", ""),
            "Start Time": exp.get("start_time", "")
        })
    return exp_data


def _render_metrics_evolution_charts(exp_df):
    """Renderiza gráficos de evolução das métricas"""
    st.subheader("📈 Evolução das Métricas")

    col1, col2 = st.columns(2)

    with col1:
        # Gráfico de Accuracy
        fig_acc = px.line(
            exp_df.reset_index(),
            x="index",
            y="Accuracy",
            title="Evolução da Acurácia",
            markers=True
        )
        fig_acc.update_layout(xaxis_title="Experimento", yaxis_title="Acurácia")
        st.plotly_chart(fig_acc, use_container_width=True)

    with col2:
        # Gráfico de F1-Score
        fig_f1 = px.line(
            exp_df.reset_index(),
            x="index",
            y="F1-Score",
            title="Evolução do F1-Score",
            markers=True,
            color_discrete_sequence=["orange"]
        )
        fig_f1.update_layout(xaxis_title="Experimento", yaxis_title="F1-Score")
        st.plotly_chart(fig_f1, use_container_width=True)


def _render_best_models_summary(exp_df):
    """Renderiza resumo dos melhores modelos"""
    st.subheader("🏆 Melhores Modelos")

    best_accuracy = exp_df.loc[exp_df['Accuracy'].idxmax()]
    best_f1 = exp_df.loc[exp_df['F1-Score'].idxmax()]

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**🎯 Melhor Acurácia:**")
        st.write(f"Run: {best_accuracy['Run ID']}")
        st.write(f"Accuracy: {best_accuracy['Accuracy']:.3f}")

    with col2:
        st.markdown("**🏅 Melhor F1-Score:**")
        st.write(f"Run: {best_f1['Run ID']}")
        st.write(f"F1-Score: {best_f1['F1-Score']:.3f}")