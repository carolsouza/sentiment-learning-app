import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def render_baseline_training_tab(api_client, api_status):
    """Renderiza a tab de Baseline Training"""
    st.header("🚀 Treinamento do Modelo Baseline")

    if 'dataset_info' in st.session_state:
        _render_training_interface(api_client, api_status)
    else:
        _render_no_dataset_message()


def _render_no_dataset_message():
    """Renderiza mensagem quando não há dataset"""
    st.warning("📁 Faça upload de um dataset primeiro para habilitar o treinamento.")
    st.info("💡 Vá para a aba 'Dataset Upload' para fazer upload do seu arquivo CSV.")

    st.markdown("---")
    st.subheader("📋 Sobre o Modelo Baseline")
    st.markdown("""
    **Naive Bayes + TF-IDF** é um modelo simples e eficaz para análise de sentimento:

    - **Naive Bayes**: Classificador probabilístico baseado no teorema de Bayes
    - **TF-IDF**: Term Frequency-Inverse Document Frequency para vetorização de texto
    - **Rápido**: Treinamento e predição muito rápidos
    - **Interpretável**: Fácil de entender e explicar
    - **Baseline**: Ponto de partida para comparar com modelos mais complexos
    """)


def _render_training_interface(api_client, api_status):
    """Renderiza a interface principal de treinamento"""
    dataset_info = st.session_state['dataset_info']

    # Informações do dataset
    st.success(f"✅ Dataset carregado: {dataset_info['filename']} ({dataset_info['size_mb']:.1f}MB)")

    # Calcula estatísticas se necessário
    _calculate_dataset_statistics(dataset_info)

    # Layout em duas colunas
    col1, col2 = st.columns([2, 1])

    with col1:
        _render_training_form(api_client, api_status)

    with col2:
        _render_training_info_panel()


def _calculate_dataset_statistics(dataset_info):
    """Calcula estatísticas do dataset se necessário"""
    if 'dataset_stats' not in st.session_state:
        with st.spinner("📊 Calculando estatísticas do dataset..."):
            try:
                df_info = dataset_info['preview_df']

                # Filtra apenas scores válidos (1,2,4,5)
                df_filtered = df_info[df_info["score"].isin([1, 2, 4, 5])].copy()
                total_samples = len(df_filtered)

                # Conta amostras por classe
                df_filtered["sentiment"] = df_filtered["score"].apply(
                    lambda x: "negativo" if x in [1, 2] else "positivo"
                )
                neg_count = len(df_filtered[df_filtered["sentiment"] == "negativo"])
                pos_count = len(df_filtered[df_filtered["sentiment"] == "positivo"])

                # Salva estatísticas no session state
                st.session_state['dataset_stats'] = {
                    'total_samples': total_samples,
                    'neg_count': neg_count,
                    'pos_count': pos_count
                }

            except Exception as e:
                st.error(f"Erro ao processar informações do dataset: {e}")
                st.session_state['dataset_stats'] = {
                    'total_samples': 10000,
                    'neg_count': 0,
                    'pos_count': 0
                }


def _render_training_form(api_client, api_status):
    """Renderiza o formulário de treinamento"""
    # Usa estatísticas já calculadas
    stats = st.session_state['dataset_stats']
    total_samples = stats['total_samples']
    neg_count = stats['neg_count']
    pos_count = stats['pos_count']

    # Informações do dataset
    st.info(f"📊 {total_samples} amostras válidas ({neg_count} negativos, {pos_count} positivos)")

    # Verifica se está treinando para evitar duplicação do formulário
    if 'training_in_progress' not in st.session_state:
        st.session_state['training_in_progress'] = False

    if not st.session_state['training_in_progress']:
        _render_training_parameters_form(api_client, api_status, total_samples, neg_count, pos_count)
    else:
        _render_training_in_progress()


def _render_training_parameters_form(api_client, api_status, total_samples, neg_count, pos_count):
    """Renderiza o formulário de parâmetros de treinamento"""
    with st.form("baseline_training_form"):
        st.subheader("⚙️ Configurações do Treinamento")

        # Seção de parâmetros do dataset
        st.markdown("**📊 Parâmetros do Dataset:**")
        max_samples = st.number_input(
            "Máximo de amostras",
            min_value=100,
            max_value=total_samples,
            value=min(50000, total_samples),
            step=1000,
            help=f"Quantidade máxima de amostras para usar no treinamento (máximo disponível: {total_samples})"
        )

        balance_data = st.checkbox(
            "Balancear classes",
            value=True,
            help="Equilibra a quantidade de amostras positivas e negativas"
        )

        # Aviso sobre balanceamento
        _render_balance_warnings(balance_data, neg_count, pos_count, max_samples)

        st.markdown("---")

        # Seção de parâmetros do modelo
        st.markdown("**🤖 Parâmetros do Modelo:**")

        col_param1, col_param2 = st.columns(2)

        with col_param1:
            test_size = st.slider(
                "Tamanho do conjunto de teste (%)",
                min_value=10,
                max_value=50,
                value=20,
                step=5,
                help="Porcentagem dos dados que será usado para teste"
            ) / 100

            max_features = st.selectbox(
                "Máximo de features TF-IDF",
                [1000, 5000, 10000, 20000],
                index=1,
                help="Número máximo de features para o vetorizador TF-IDF"
            )

        with col_param2:
            experiment_name = st.text_input(
                "Nome do Experimento",
                value="sentiment_analysis_baseline",
                help="Nome para organizar os experimentos no MLflow"
            )

            # Mostra distribuição final
            if balance_data and neg_count > 0 and pos_count > 0:
                min_class = min(neg_count, pos_count)
                final_samples = min(min_class * 2, max_samples)
                st.metric("Amostras finais estimadas", final_samples)
            else:
                final_samples = min(total_samples, max_samples)
                st.metric("Amostras finais estimadas", final_samples)

        st.markdown("---")

        # Botão de treinamento
        col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
        with col_btn2:
            submit_training = st.form_submit_button(
                "🚀 Iniciar Treinamento Baseline",
                type="primary",
                disabled=not api_status,
                use_container_width=True
            )

        if submit_training:
            _handle_training_submission(
                api_client, api_status, max_samples, balance_data,
                test_size, max_features, experiment_name
            )


def _render_balance_warnings(balance_data, neg_count, pos_count, max_samples):
    """Renderiza avisos sobre balanceamento"""
    if balance_data and neg_count > 0 and pos_count > 0:
        min_class = min(neg_count, pos_count)
        max_balanced_samples = min_class * 2

        if max_samples > max_balanced_samples:
            st.warning(f"⚠️ Com balanceamento, máximo utilizável: {max_balanced_samples} amostras "
                     f"({min_class} de cada classe)")

    elif balance_data and (neg_count == 0 or pos_count == 0):
        st.error("❌ Não é possível balancear: apenas uma classe disponível")
        st.info("💡 Desmarque 'Balancear classes' para usar todos os dados")


def _render_training_in_progress():
    """Renderiza interface durante o treinamento"""
    st.info("🤖 Treinamento em progresso...")
    st.warning("⏳ Aguarde a conclusão do treinamento atual.")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Atualizar Status", use_container_width=True):
            st.session_state['training_in_progress'] = False
            st.rerun()

    with col2:
        if st.button("❌ Cancelar Treinamento", use_container_width=True):
            st.session_state['training_in_progress'] = False
            st.warning("Treinamento cancelado pelo usuário")
            st.rerun()


def _render_training_info_panel():
    """Renderiza painel de informações sobre o treinamento"""
    st.subheader("📋 Sobre o Treinamento")

    st.markdown("""
    **Processo de Treinamento:**

    1. **Preparação**: Filtrar e balancear dados
    2. **Vetorização**: Converter texto em features TF-IDF
    3. **Divisão**: Separar treino/teste
    4. **Treinamento**: Treinar Naive Bayes
    5. **Avaliação**: Calcular métricas
    6. **Registro**: Salvar no MLflow
    """)

    st.markdown("---")

    # Últimos resultados
    if 'last_training_result' in st.session_state:
        st.markdown("**🏆 Último Treinamento:**")
        result = st.session_state['last_training_result']

        if result.get("success", False):
            metrics = result.get("metrics", {})
            st.success(f"✅ Acurácia: {metrics.get('accuracy', 0):.3f}")
            st.info(f"🆔 Run: {result.get('run_id', '')}")
        else:
            st.error("❌ Último treinamento falhou")


def _handle_training_submission(api_client, api_status, max_samples, balance_data, test_size, max_features, experiment_name):
    """Manipula o envio do formulário de treinamento"""
    # Marca como treinando para evitar duplicação do formulário
    st.session_state['training_in_progress'] = True

    if not api_status:
        st.error("API não está disponível. Verifique se o serviço está rodando.")
        st.session_state['training_in_progress'] = False
    else:
        with st.spinner("🤖 Treinando modelo... Isso pode levar alguns minutos."):
            # Prepara request para a API usando o file_path da API
            training_request = {
                "dataset_path": st.session_state['dataset_info']['file_path'],
                "max_samples": max_samples,
                "balance_data": balance_data,
                "test_size": test_size,
                "max_features": max_features,
                "experiment_name": experiment_name
            }

            # Chama a API
            result = api_client.train_model(training_request)

        # Exibe resultados
        _display_training_results(result)

        # Marca treinamento como concluído
        st.session_state['training_in_progress'] = False


def _display_training_results(result):
    """Exibe os resultados do treinamento"""
    if result.get("success", False):
        st.success("✅ Modelo treinado com sucesso!")

        # Salva resultado na sessão
        st.session_state['last_training_result'] = result

        # Exibe métricas em cards
        st.subheader("📊 Resultados do Treinamento")

        metrics = result.get("metrics", {})
        col_metric1, col_metric2, col_metric3, col_metric4 = st.columns(4)

        with col_metric1:
            st.metric("🎯 Acurácia", f"{metrics.get('accuracy', 0):.3f}")
        with col_metric2:
            st.metric("📈 F1-Score", f"{metrics.get('f1_score', 0):.3f}")
        with col_metric3:
            st.metric("🔍 Precisão", f"{metrics.get('precision', 0):.3f}")
        with col_metric4:
            st.metric("📊 Recall", f"{metrics.get('recall', 0):.3f}")
            
        # Informações sobre balanceamento
        if 'balancing_info' in result:
            _render_balancing_info(result['balancing_info'])

        # Gráficos detalhados dos resultados
        _render_detailed_metrics_charts(metrics, result)

        # Informações do experimento
        st.info(f"📋 Experimento ID: {result.get('experiment_id', '')}")
        st.info(f"🏃 Run ID: {result.get('run_id', '')}")

    else:
        st.error(f"❌ Erro no treinamento: {result.get('message', 'Erro desconhecido')}")

        # Sugestões para resolução
        st.markdown("**💡 Possíveis soluções:**")
        st.write("• Verifique se a API está rodando")
        st.write("• Confirme se o dataset está válido")
        st.write("• Tente reduzir o número de amostras")
        st.write("• Verifique os logs da API para mais detalhes")


def _render_detailed_metrics_charts(metrics, result):
    """Renderiza gráficos detalhados das métricas do modelo"""
    st.markdown("---")
    st.subheader("📈 Análise Detalhada do Modelo")

    # Cria tabs para diferentes visualizações
    chart_tab1, chart_tab2, chart_tab3 = st.tabs([
        "📊 Métricas Gerais",
        "🎯 Matriz de Confusão",
        "📈 Performance por Classe"
    ])

    with chart_tab1:
        _render_general_metrics_chart(metrics)

    with chart_tab2:
        _render_confusion_matrix_chart(result)

    with chart_tab3:
        _render_class_performance_chart(result)


def _render_general_metrics_chart(metrics):
    """Gráfico de radar com as métricas gerais"""
    st.markdown("**🎯 Visão Geral das Métricas**")

    # Dados para o gráfico de radar
    metrics_names = ['Acurácia', 'Precisão', 'Recall', 'F1-Score']
    metrics_values = [
        metrics.get('accuracy', 0),
        metrics.get('precision', 0),
        metrics.get('recall', 0),
        metrics.get('f1_score', 0)
    ]

    # Gráfico de radar
    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=metrics_values,
        theta=metrics_names,
        fill='toself',
        name='Modelo Baseline',
        line_color='blue',
        fillcolor='rgba(0, 100, 255, 0.1)'
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=True,
        title="Radar de Performance do Modelo",
        height=400
    )

    st.plotly_chart(fig, use_container_width=True)

    # Interpretação das métricas
    col1, col2 = st.columns(2)
    with col1:
        st.info("**📖 Interpretação:**")
        st.write(f"• **Acurácia**: {metrics.get('accuracy', 0):.1%} de predições corretas")
        st.write(f"• **Precisão**: {metrics.get('precision', 0):.1%} dos positivos preditos são corretos")

    with col2:
        st.write(f"• **Recall**: {metrics.get('recall', 0):.1%} dos positivos reais foram encontrados")
        st.write(f"• **F1-Score**: {metrics.get('f1_score', 0):.1%} harmonia entre precisão e recall")


def _render_confusion_matrix_chart(result):
    """Matriz de confusão com dados reais do MLflow"""
    st.markdown("**🎯 Matriz de Confusão**")

    # Usa dados reais da matriz de confusão
    if 'confusion_matrix' in result:
        cm = result['confusion_matrix']
    else:
        st.warning("⚠️ Dados da matriz de confusão não disponíveis")
        return

    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=['Predito: Negativo', 'Predito: Positivo'],
        y=['Real: Negativo', 'Real: Positivo'],
        colorscale='Blues',
        text=cm,
        texttemplate="%{text}",
        textfont={"size": 20},
        showscale=True
    ))

    fig.update_layout(
        title="Matriz de Confusão",
        xaxis_title="Predição",
        yaxis_title="Valor Real",
        height=400
    )

    st.plotly_chart(fig, use_container_width=True)

    # Explicação da matriz
    st.info("""
    **📖 Como interpretar:**
    - **Diagonal principal**: Predições corretas
    - **Diagonal secundária**: Erros do modelo
    - **Valores maiores na diagonal = melhor modelo**
    """)


def _render_class_performance_chart(result):
    """Performance por classe com dados reais do MLflow"""
    st.markdown("**📈 Performance por Classe**")

    # Extrai métricas por classe dos dados reais
    if 'class_metrics' in result:
        class_report = result['class_metrics']

        # Extrai apenas as classes (remove macro avg, weighted avg, etc)
        class_data = {}
        for class_name, class_metrics in class_report.items():
            if isinstance(class_metrics, dict) and class_name not in ['accuracy', 'macro avg', 'weighted avg']:
                class_data[class_name] = {
                    'precision': class_metrics.get('precision', 0),
                    'recall': class_metrics.get('recall', 0),
                    'f1_score': class_metrics.get('f1-score', 0)
                }
    else:
        st.warning("⚠️ Dados de métricas por classe não disponíveis")
        return

    # Prepara dados para o gráfico
    classes = list(class_data.keys())
    precision_vals = [class_data[cls]['precision'] for cls in classes]
    recall_vals = [class_data[cls]['recall'] for cls in classes]
    f1_vals = [class_data[cls]['f1_score'] for cls in classes]

    # Gráfico de barras agrupadas
    fig = go.Figure()

    fig.add_trace(go.Bar(
        name='Precisão',
        x=classes,
        y=precision_vals,
        marker_color='lightblue'
    ))

    fig.add_trace(go.Bar(
        name='Recall',
        x=classes,
        y=recall_vals,
        marker_color='lightgreen'
    ))

    fig.add_trace(go.Bar(
        name='F1-Score',
        x=classes,
        y=f1_vals,
        marker_color='lightcoral'
    ))

    fig.update_layout(
        title='Métricas por Classe',
        xaxis_title='Classe',
        yaxis_title='Score',
        barmode='group',
        height=400,
        yaxis=dict(range=[0, 1])
    )

    st.plotly_chart(fig, use_container_width=True)

    # Tabela com valores exatos
    df_class = pd.DataFrame(class_data).T
    df_class = df_class.round(3)
    st.dataframe(df_class, use_container_width=True)


def _render_balancing_info(balancing_info):

    was_balanced = balancing_info.get('was_balanced', False)
    neg_count = balancing_info.get('final_negative_count', 0)
    pos_count = balancing_info.get('final_positive_count', 0)
    is_perfectly_balanced = balancing_info.get('is_perfectly_balanced', False)

    col1, col2 = st.columns(2)

    with col1:
        st.metric("➖ Amostras Negativas", neg_count)

    with col2:
        st.metric("➕ Amostras Positivas", pos_count)

    # Status do balanceamento
    if was_balanced:
        if is_perfectly_balanced:
            st.success("✅ **Balanceamento Perfeito**: Dataset perfeitamente balanceado 50/50!")
        else:
            ratio = max(neg_count, pos_count) / max(min(neg_count, pos_count), 1)
            st.warning(f"⚠️ **Balanceamento Limitado**: Proporção {ratio:.1f}:1. "
                      f"Pode ser devido à limitação de amostras disponíveis em uma das classes.")

    else:
        st.info("ℹ️ **Sem Balanceamento**: Dataset usado na proporção original.")

        total = neg_count + pos_count
        neg_percent = (neg_count / total) * 100 if total > 0 else 0
        pos_percent = (pos_count / total) * 100 if total > 0 else 0

        st.write(f"• **Negativos**: {neg_count} amostras ({neg_percent:.1f}%)")
        st.write(f"• **Positivos**: {pos_count} amostras ({pos_percent:.1f}%)")
