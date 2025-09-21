import streamlit as st
import pandas as pd
import os
import plotly.express as px
from datetime import datetime


def render_dataset_training_tab(api_client, api_status):
    """Renderiza a tab de Dataset Upload"""
    st.header("📊 Dataset Upload")

    _render_dataset_upload_section(api_client, api_status)


def _render_dataset_upload_section(api_client, api_status):
    """Seção de upload de dataset"""
    st.subheader("📁 Dataset Upload")
    uploaded_file = st.file_uploader(
        "Escolha o arquivo CSV do Amazon Fine Foods Reviews",
        type=["csv"],
        help="O arquivo deve conter as colunas 'score' (1-5) e 'text'",
        accept_multiple_files=False
    )

    if uploaded_file is not None:
        # Verifica se é um arquivo novo ou já processado
        current_file_id = f"{uploaded_file.name}_{uploaded_file.size}"

        if 'last_uploaded_file' not in st.session_state or st.session_state['last_uploaded_file'] != current_file_id:
            # Novo arquivo - processa e envia
            try:
                _process_new_file(uploaded_file, api_client, api_status, current_file_id)
            except pd.errors.EmptyDataError:
                st.error("❌ Arquivo CSV está vazio ou mal formatado")
            except pd.errors.ParserError as e:
                st.error(f"❌ Erro ao processar CSV: {str(e)}")
            except MemoryError:
                st.error("❌ Arquivo muito grande para processar. Tente um arquivo menor.")
            except Exception as e:
                st.error(f"❌ Erro inesperado: {str(e)}")
                st.error("Verifique se o arquivo é um CSV válido com cabeçalhos")

        else:
            # Arquivo já processado - mostra informações do cache
            if 'dataset_info' in st.session_state:
                dataset_info = st.session_state['dataset_info']
                st.info(f"📁 Arquivo já enviado: {dataset_info['filename']} ({dataset_info['size_mb']:.1f}MB)")
                st.success("✅ Usando arquivo já carregado na API")

    # Se nenhum arquivo foi carregado mas há dataset em cache
    elif 'dataset_info' in st.session_state:
        # Mostra informações do dataset já carregado
        dataset_info = st.session_state['dataset_info']
        st.info(f"📁 Dataset já carregado: {dataset_info['filename']} ({dataset_info['size_mb']:.1f}MB)")
        st.success("✅ Pronto para treinamento!")

    # Exibe preview dos dados se houver DataFrame disponível e não estiver processando
    if 'dataset_info' in st.session_state and 'last_uploaded_file' in st.session_state:
        _render_dataset_preview()


def _process_new_file(uploaded_file, api_client, api_status, current_file_id):
    """Processa um novo arquivo"""
    # Verifica tamanho do arquivo
    file_size_mb = uploaded_file.size / (1024 * 1024)

    if file_size_mb > 500:
        st.error(f"❌ Arquivo muito grande: {file_size_mb:.1f}MB. Máximo permitido: 500MB")
    else:
        st.info(f"📁 Novo arquivo: {uploaded_file.name} ({file_size_mb:.1f}MB)")

        # Progress bar para processamento
        progress_bar = st.progress(0)
        status_text = st.empty()

        status_text.text("📖 Lendo arquivo CSV...")
        progress_bar.progress(25)

        # Lê arquivo para preview (sem salvar localmente)
        file_content = uploaded_file.getvalue()
        df = pd.read_csv(uploaded_file)

        progress_bar.progress(50)
        status_text.text("✅ Validando dados...")

        # Normaliza headers para lowercase para exibição
        df.columns = df.columns.str.lower()

        progress_bar.progress(75)
        status_text.text("📤 Enviando para API...")

        # Envia arquivo para a API
        if api_status:
            try:
                upload_result = api_client.upload_dataset(file_content, uploaded_file.name)
            except AttributeError as e:
                st.error(f"Erro no cliente da API: {str(e)}")
                st.info("Tentando recarregar a página pode resolver o problema")
                upload_result = {"success": False, "message": "Erro no cliente da API"}

            if upload_result.get("success", False):
                progress_bar.progress(100)
                status_text.text("✅ Upload concluído!")

                # Armazena informações do arquivo na sessão
                st.session_state['dataset_info'] = {
                    'filename': upload_result['filename'],
                    'file_path': upload_result['file_path'],
                    'size_mb': upload_result['size_mb'],
                    'preview_df': df  # DataFrame para preview
                }

                # Marca arquivo como processado
                st.session_state['last_uploaded_file'] = current_file_id

                # Limpa estatísticas antigas para forçar recálculo
                if 'dataset_stats' in st.session_state:
                    del st.session_state['dataset_stats']

                # Limpa elementos de UI temporários
                progress_bar.empty()
                status_text.empty()

                st.success(f"✅ Arquivo enviado para API: {upload_result['filename']}")

                # Força rerun para limpar interface
                st.rerun()
            else:
                progress_bar.progress(0)
                status_text.text("❌ Erro no upload")
                st.error(f"Erro no upload: {upload_result['message']}")
        else:
            progress_bar.progress(0)
            status_text.text("❌ API offline")
            st.error("API não está disponível para receber o arquivo")


def _render_dataset_preview():
    """Renderiza o preview do dataset"""
    dataset_info = st.session_state['dataset_info']
    df = dataset_info['preview_df']

    # Validação básica
    required_cols = ['score', 'text']
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        st.error(f"Colunas ausentes: {', '.join(missing_cols)}")
    else:
        st.success("✅ Dataset válido!")

        # Preview dos dados
        st.subheader("👀 Preview dos Dados")
        st.dataframe(df.head(10), use_container_width=True)

        # Estatísticas básicas
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("Total de Registros", len(df))
        with col_b:
            st.metric("Registros Válidos", len(df.dropna(subset=['score', 'text'])))
        with col_c:
            valid_scores = df[df['score'].isin([1, 2, 4, 5])]
            st.metric("Para Análise (scores 1,2,4,5)", len(valid_scores))

        # Distribuição dos scores
        st.subheader("📈 Distribuição dos Scores")
        score_counts = df['score'].value_counts().sort_index()

        fig = px.bar(
            x=score_counts.index,
            y=score_counts.values,
            labels={'x': 'Score', 'y': 'Quantidade'},
            title="Distribuição dos Scores no Dataset"
        )
        st.plotly_chart(fig, use_container_width=True)


