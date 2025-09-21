import requests
import json
from typing import Dict, Any, List, Optional


class MLAPIClient:
    def __init__(self, api_base_url: str = "http://localhost:8000"):
        self.api_base_url = api_base_url.rstrip("/")

    def health_check(self) -> bool:
        """Verifica se a API está funcionando"""
        try:
            response = requests.get(f"{self.api_base_url}/health", timeout=5)
            return response.status_code == 200
        except:
            return False

    def upload_dataset(self, file_content: bytes, filename: str) -> Dict[str, Any]:
        """Upload de dataset para a API"""
        try:
            files = {"file": (filename, file_content, "text/csv")}
            response = requests.post(
                f"{self.api_base_url}/upload",
                files=files,
                timeout=120  # 2 minutos para upload
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {
                "success": False,
                "message": f"Erro no upload: {str(e)}",
                "filename": "",
                "file_path": "",
                "size_mb": 0
            }

    def train_model(self, training_request: Dict[str, Any]) -> Dict[str, Any]:
        """Inicia o treinamento de um modelo"""
        try:
            response = requests.post(
                f"{self.api_base_url}/train",
                json=training_request,
                timeout=300  # 5 minutos timeout para treinamento
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {
                "success": False,
                "message": f"Erro na comunicação com a API: {str(e)}",
                "experiment_id": "",
                "run_id": "",
                "metrics": {},
                "model_uri": ""
            }

    def predict(self, text: str, model_uri: str) -> Dict[str, Any]:
        """Faz predição usando um modelo treinado"""
        try:
            response = requests.post(
                f"{self.api_base_url}/predict",
                json={"text": text, "model_uri": model_uri},
                timeout=30
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {
                "error": f"Erro na predição: {str(e)}"
            }

