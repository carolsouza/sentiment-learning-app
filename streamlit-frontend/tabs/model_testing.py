import streamlit as st
import pandas as pd
import plotly.express as px


def render_model_testing_tab(api_client, api_status):
    """Renderiza a tab de Model Testing"""
    st.header("🧪 Teste de Modelos")

    if 'last_training_result' in st.session_state:
        _render_model_testing_interface(api_client, api_status)
    else:
        st.info("🤖 Execute um treinamento primeiro para testar o modelo.")


def _render_model_testing_interface(api_client, api_status):
    """Renderiza a interface de teste do modelo"""
    result = st.session_state['last_training_result']
    model_uri = result.get('model_uri', '')

    if model_uri and api_status:
        st.success(f"✅ Modelo carregado: {result.get('run_id', '')}")

        _render_prediction_form(api_client, model_uri)
        _render_test_examples()
    else:
        st.warning("⚠️ Modelo não disponível ou API offline.")


def _render_prediction_form(api_client, model_uri):
    """Renderiza o formulário de predição"""
    # Formulário de teste
    with st.form("prediction_form"):
        st.subheader("✍️ Digite um texto para análise:")
        test_text = st.text_area(
            "Texto:",
            placeholder="Ex: This product is amazing! I really love it.",
            height=100,
            help="Digite qualquer texto em inglês para análise de sentimento"
        )

        submit_prediction = st.form_submit_button(
            "🎯 Analisar Sentimento",
            type="primary"
        )

        if submit_prediction and test_text:
            _handle_prediction_request(api_client, model_uri, test_text)


def _handle_prediction_request(api_client, model_uri, test_text):
    """Manipula a requisição de predição"""
    with st.spinner("🤖 Analisando sentimento..."):
        prediction_result = api_client.predict(test_text, model_uri)

        if "error" not in prediction_result:
            _display_prediction_results(prediction_result)
        else:
            st.error(f"Erro na predição: {prediction_result['error']}")


def _display_prediction_results(prediction_result):
    """Exibe os resultados da predição"""
    prediction = prediction_result.get("prediction", "")
    probabilities = prediction_result.get("probabilities", {})

    # Exibe resultado
    st.subheader("🎯 Resultado da Análise")

    # Determina cor baseada na predição
    color = "green" if prediction == "positivo" else "red"
    st.markdown(f"**Sentimento Detectado:** <span style='color: {color}; font-size: 1.5em;'>{prediction.upper()}</span>", unsafe_allow_html=True)

    # Gráfico de probabilidades
    if probabilities:
        _render_probability_chart(probabilities)
        _render_probability_table(probabilities)


def _render_probability_chart(probabilities):
    """Renderiza gráfico de probabilidades"""
    st.subheader("📊 Probabilidades")

    prob_df = pd.DataFrame(
        list(probabilities.items()),
        columns=['Sentimento', 'Probabilidade']
    )

    fig = px.bar(
        prob_df,
        x='Sentimento',
        y='Probabilidade',
        color='Sentimento',
        color_discrete_map={'positivo': 'green', 'negativo': 'red'},
        title="Distribuição de Probabilidades"
    )
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)


def _render_probability_table(probabilities):
    """Renderiza tabela de probabilidades"""
    prob_df = pd.DataFrame(
        list(probabilities.items()),
        columns=['Sentimento', 'Probabilidade']
    )
    st.dataframe(prob_df, use_container_width=True)


def _render_test_examples():
    """Renderiza exemplos de teste"""
    st.subheader("💡 Exemplos para Testar")
    examples = [
        "This product is absolutely amazing! Best purchase ever.",
        "Terrible quality, completely disappointed with this item.",
        "The food was okay, nothing special but not bad either.",
        "Outstanding service and excellent quality. Highly recommended!",
        "Worst experience ever. Will never buy again."
    ]

    for i, example in enumerate(examples):
        if st.button(f"Exemplo {i+1}: {example[:50]}...", key=f"example_{i}"):
            st.session_state['example_text'] = example