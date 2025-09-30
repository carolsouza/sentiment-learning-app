import streamlit as st


def render_model_testing_tab(api_client, api_status):
    """Renderiza a tab de Model Testing - Teste simples com modelo de produção"""
    st.header("🧪 Teste de Predição")

    if not api_status:
        st.error("🚫 API não está disponível")
        st.info("💡 Verifique se a API está online e se a API Key está configurada")
        return

    st.info("🚀 **Modelo de Produção**: Testando com o modelo BiLSTM otimizado")

    # Input simples de texto
    user_text = st.text_area(
        "Digite um texto para analisar:",
        placeholder="Ex: O atendimento foi excepcional e o produto chegou rapidamente!",
        height=100
    )

    if st.button("🚀 Analisar Sentimento", use_container_width=True):
        if user_text.strip():
            with st.spinner("Analisando com modelo de produção..."):
                result = api_client.predict_production(user_text.strip())

            if result["success"]:
                prediction = result["prediction"]
                _display_result(prediction)
            else:
                st.error(f"❌ Erro: {result.get('error', 'Erro desconhecido')}")
        else:
            st.warning("⚠️ Digite um texto para análise")


def _display_result(prediction):
    """Exibe resultado da predição"""
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