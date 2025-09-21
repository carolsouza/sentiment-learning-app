from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import logging
import os
from typing import List
from datetime import datetime

from .models import (
    TrainingRequest,
    TrainingResponse,
    PredictionRequest,
    PredictionResponse
)
from .ml_service import MLService

# Configuração de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Inicializa FastAPI
app = FastAPI(
    title="ML Training API",
    description="API para treinamento de modelos de análise de sentimento",
    version="1.0.0"
)

# Configuração CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Em produção, especificar domínios específicos
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inicializa o serviço ML
ml_service = MLService()

# Estado global para tracking de jobs
training_jobs = {}

# Cria diretório para datasets se não existir
os.makedirs("datasets", exist_ok=True)


@app.get("/")
async def root():
    return {"message": "ML Training API is running"}


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


@app.post("/upload")
async def upload_dataset(file: UploadFile = File(...)):
    """Upload de dataset CSV"""
    try:
        # Verifica se é um arquivo CSV
        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="Arquivo deve ser CSV")

        # Verifica tamanho (500MB = 500 * 1024 * 1024 bytes)
        max_size = 500 * 1024 * 1024
        content = await file.read()

        if len(content) > max_size:
            raise HTTPException(status_code=400, detail="Arquivo muito grande (máximo 500MB)")

        # Salva arquivo com timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"dataset_{timestamp}.csv"
        file_path = os.path.join("datasets", filename)

        with open(file_path, "wb") as f:
            f.write(content)

        logger.info(f"Dataset uploaded: {filename} ({len(content)} bytes)")

        return {
            "success": True,
            "message": "Dataset uploaded successfully",
            "filename": filename,
            "file_path": file_path,
            "size_mb": len(content) / (1024 * 1024)
        }

    except Exception as e:
        logger.error(f"Upload failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erro no upload: {str(e)}")


@app.post("/train", response_model=TrainingResponse)
async def train_model(request: TrainingRequest, background_tasks: BackgroundTasks):
    """Inicia o treinamento de um modelo"""
    try:
        logger.info(f"Starting training with request: {request}")

        # Verifica se o arquivo existe
        if not os.path.exists(request.dataset_path):
            raise HTTPException(status_code=400, detail=f"Dataset não encontrado: {request.dataset_path}")

        # Inicia o treinamento
        result = ml_service.train_model(
            dataset_path=request.dataset_path,
            max_samples=request.max_samples,
            balance_data=request.balance_data,
            test_size=request.test_size,
            max_features=request.max_features,
            experiment_name=request.experiment_name
        )

        if result["success"]:
            return TrainingResponse(**result)
        else:
            raise HTTPException(status_code=500, detail=result["message"])

    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erro no treinamento: {str(e)}")


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Faz predição usando um modelo treinado"""
    try:
        result = ml_service.predict(request.text, request.model_uri)
        return PredictionResponse(**result)

    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erro na predição: {str(e)}")




if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)