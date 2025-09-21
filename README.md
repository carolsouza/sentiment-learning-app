# 🤖 Sentiment Analysis ML Platform

Uma plataforma completa para treinamento e análise de modelos de sentiment analysis usando Naive Bayes + TF-IDF, com tracking de experimentos via MLflow.

## 📋 Arquitetura do Sistema

```
sentiment-learning-app/
├── streamlit-frontend/    # Interface web principal (Streamlit)
├── ml-api/                # API de treinamento (FastAPI)
├── mlflow-tracking/       # MLflow tracking e artifacts
```

## 🚀 Componentes

### 1. **Streamlit Frontend** (Port 8501)
- Interface amigável para upload de datasets
- Configuração de parâmetros de treinamento
- Visualização de resultados e métricas
- Teste de modelos treinados
- Dashboard de experimentos

### 2. **FastAPI Backend** (Port 8000)
- API REST para treinamento de modelos
- Integração com MLflow para tracking
- Endpoints para predição
- Gestão de experimentos

### 3. **MLflow Tracking** (Port 5000)
- Tracking de experimentos
- Versionamento de modelos
- Métricas e artifacts
- Comparação de resultados

## 📦 Instalação

### Pré-requisitos
```bash
Python 3.8+
pip
```
Recomendado utilização de virtualização, como o venv por exemplo.

### Setup Completo
```bash
# 1. Clone o repositório
git clone <repo-url>
cd sentiment-learning-app

# 2. Instale todas as dependências automaticamente (Windows)
install_dependencies.bat

# 3. Ou instale manualmente:
cd ml-api && pip install -r requirements.txt && cd ..
cd streamlit-frontend && pip install -r requirements.txt && cd ..
cd mlflow-tracking && pip install -r requirements.txt && cd ..
```

## 🎯 Como Usar

### Opção 1: Inicialização Automática (Windows)
```bash
# Inicia todos os serviços automaticamente
start_all.bat
```

### Opção 2: Inicialização Manual

#### 1. Iniciar MLflow Server
```bash
cd mlflow-tracking
python start_mlflow.py
# Acesse: http://localhost:5000
```

#### 2. Iniciar FastAPI Backend
```bash
cd ml-api
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
# API Docs: http://localhost:8000/docs
```

#### 3. Iniciar Streamlit Frontend
```bash
cd streamlit-frontend
streamlit run app.py --server.port 8501
# Interface: http://localhost:8501
```

## 📊 Workflow de Uso

1. **Upload Dataset**: Faça upload do dataset Amazon Fine Foods Reviews
2. **Configuração**: Ajuste parâmetros como max_samples, balance_data, test_size
3. **Treinamento**: Clique em "Treinar Modelo Baseline"
4. **Teste**: Teste o modelo com textos customizados

## 🔧 Endpoints da API

### Treinamento
```http
POST /train
Content-Type: application/json

{
  "dataset_path": "path/to/dataset.csv",
  "max_samples": 5000,
  "balance_data": true,
  "test_size": 0.2,
  "max_features": 5000,
  "experiment_name": "sentiment_analysis"
}
```

### Predição
```http
POST /predict
Content-Type: application/json

{
  "text": "This product is amazing!",
  "model_uri": "runs:/run_id/model"
}
```

### Experimentos
```http
GET /experiments
GET /experiments/{experiment_name}
```

## 📈 Métricas Tracked

- **Accuracy**: Acurácia geral do modelo
- **F1-Score**: Média harmônica entre precisão e recall
- **Precision**: Precisão por classe
- **Recall**: Revocação por classe
- **Confusion Matrix**: Matriz de confusão
- **Classification Report**: Relatório detalhado por classe

## 🗂️ Formato do Dataset

O dataset deve conter as colunas:
- **score**: Valores de 1-5 (1,2=negativo, 4,5=positivo, 3=ignorado)
- **text**: Texto dos reviews para análise

Exemplo:
```csv
score,text
5,"This product is amazing! Love it."
1,"Terrible quality, very disappointed."
4,"Good value for money."
2,"Not what I expected."
```

## ⚙️ Configurações do Modelo

### Naive Bayes + TF-IDF
- **Algoritmo**: MultinomialNB com alpha=1.0
- **Vetorização**: TF-IDF com n-gramas (1,2)
- **Stop Words**: Inglês
- **Min DF**: 2
- **Max Features**: Configurável (1K-20K)

### Parâmetros Ajustáveis
- **max_samples**: Limite de amostras para treinamento
- **balance_data**: Balanceamento automático de classes
- **test_size**: Porcentagem para conjunto de teste
- **max_features**: Máximo de features TF-IDF

## 🚀 API Endpoints

### Base URL: `http://localhost:8000`

#### 1. Health Check
```http
GET /health
```
**Resposta:**
```json
{
  "status": "healthy"
}
```

#### 2. Upload de Dataset
```http
POST /upload
Content-Type: multipart/form-data
```
**Parâmetros:**
- `file`: Arquivo CSV (max 500MB)

**Resposta:**
```json
{
  "success": true,
  "message": "Dataset uploaded successfully",
  "filename": "dataset_20241221_143022.csv",
  "file_path": "datasets/dataset_20241221_143022.csv",
  "size_mb": 45.2
}
```

#### 3. Treinamento de Modelo
```http
POST /train
Content-Type: application/json
```
**Body:**
```json
{
  "dataset_path": "datasets/dataset_20241221_143022.csv",
  "max_samples": 50000,
  "balance_data": true,
  "test_size": 0.2,
  "max_features": 10000,
  "experiment_name": "sentiment_analysis_baseline"
}
```

**Resposta:**
```json
{
  "success": true,
  "message": "Modelo treinado com sucesso",
  "experiment_id": "780504671289068133",
  "run_id": "2295a696ba2145ad94591b9a3bbbb414",
  "metrics": {
    "accuracy": 0.876,
    "f1_score": 0.874,
    "precision": 0.878,
    "recall": 0.872
  },
  "confusion_matrix": [[8420, 1580], [1245, 8755]],
  "class_metrics": {
    "negativo": {"precision": 0.871, "recall": 0.842, "f1-score": 0.856},
    "positivo": {"precision": 0.847, "recall": 0.876, "f1-score": 0.861}
  },
  "balancing_info": {
    "was_balanced": true,
    "final_negative_count": 10000,
    "final_positive_count": 10000,
    "is_perfectly_balanced": true
  },
  "model_uri": "runs:/2295a696ba2145ad94591b9a3bbbb414/model"
}
```

#### 4. Predição
```http
POST /predict
Content-Type: application/json
```
**Body:**
```json
{
  "text": "This product is amazing! I really love it.",
  "model_uri": "runs:/2295a696ba2145ad94591b9a3bbbb414/model"
}
```

**Resposta:**
```json
{
  "prediction": "positivo",
  "probabilities": {
    "negativo": 0.123,
    "positivo": 0.877
  }
}
```

## 🛠️ Desenvolvimento

### Estrutura de Arquivos

#### ML API
```
ml-api/
├── app/
│   ├── __init__.py
│   ├── main.py          # FastAPI app
│   └── ml_service.py    # ML training logic
├── datasets/            # Uploaded datasets
└── requirements.txt
```

#### Streamlit Frontend
```
streamlit-frontend/
├── tabs/
│   ├── __init__.py
│   ├── api_status.py        # API status tab
│   ├── baseline_training.py # Baseline training tab
│   ├── dataset_training.py  # Dataset upload tab
│   ├── experiments.py       # Experiments tab
│   └── model_testing.py     # Model testing tab
├── app.py                   # Main Streamlit app
├── api_client.py            # API client
└── requirements.txt
```

#### MLflow
```
mlflow-tracking/
├── start_mlflow.py     # MLflow server setup
├── mlruns/             # MLflow experiments data
├── mlartifacts/        # MLflow artifacts storage
└── requirements.txt
```

#### Scripts de Inicialização
```
├── start_all.bat           # Inicia todos os serviços (Windows)
└── install_dependencies.bat # Instala todas as dependências
```
