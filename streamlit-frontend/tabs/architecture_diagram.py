import streamlit as st
import os


def render_architecture_diagram_tab(api_client, api_status):
    """Renderiza a tab de Diagrama da Arquitetura"""
    st.header("🏗️ Arquitetura do Sistema")

    # CSS para ajustar espaçamento e centralizar
    st.markdown("""
    <style>
    .main .block-container {
        padding-top: 1rem !important;
    }
    h1, h2, h3 {
        margin-bottom: 0.5rem !important;
    }
    </style>
    """, unsafe_allow_html=True)

    # Caminho para a imagem do diagrama
    image_path = "assets/arquitetura_projeto.png"

    # Verificar se a imagem existe
    if os.path.exists(image_path):
        # Centralizar e limitar o tamanho da imagem
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(
                image_path,
                caption="Arquitetura do Sistema de Sentiment Analysis",
                use_container_width=True
            )
    else:
        st.error("Imagem do diagrama não encontrada em assets/arquitetura_projeto.png")