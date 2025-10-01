import requests
import json
import os
import urllib.parse
from typing import Dict, Any, List, Optional

# Tenta carregar arquivo .env se disponível
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # python-dotenv não está instalado, continua sem carregar .env
    pass


class DeepLearningAPIClient:
    def __init__(self, api_base_url: str = None, api_key: str = None):
        # Prioridade: parâmetro > variável de ambiente > padrão GCP
        self.api_base_url = (
            api_base_url or
            os.getenv("API_BASE_URL") or
            "https://deeplearning-infnet-project-api-966997855008.europe-west1.run.app"
        ).rstrip("/")

        # Tenta pegar API_KEY do ambiente se não fornecida
        self.api_key = api_key or os.getenv("API_KEY") or os.getenv("DEEPLEARNING_API_KEY")
        self.headers = {"X-API-Key": self.api_key} if self.api_key else {}

    def has_api_key(self) -> bool:
        """Verifica se API key está configurada"""
        return bool(self.api_key)

    def health_check(self) -> Dict[str, Any]:
        """Verifica se a API está funcionando"""
        try:
            if not self.has_api_key():
                return {
                    "success": False,
                    "status": "api_key_missing",
                    "message": "API Key não configurada. Configure a variável de ambiente API_KEY ou DEEPLEARNING_API_KEY"
                }

            response = requests.get(
                f"{self.api_base_url}/health",
                headers=self.headers,
                timeout=10
            )

            if response.status_code == 200:
                return {
                    "success": True,
                    "status": "online",
                    "data": response.json(),
                    "message": "API está funcionando corretamente"
                }
            elif response.status_code == 401:
                return {
                    "success": False,
                    "status": "unauthorized",
                    "message": "API Key inválida ou não autorizada"
                }
            else:
                return {
                    "success": False,
                    "status": "error",
                    "message": f"API retornou status {response.status_code}"
                }
        except requests.exceptions.Timeout:
            return {
                "success": False,
                "status": "timeout",
                "message": "Timeout ao conectar com a API"
            }
        except requests.exceptions.ConnectionError:
            return {
                "success": False,
                "status": "connection_error",
                "message": "Erro de conexão com a API"
            }
        except Exception as e:
            return {
                "success": False,
                "status": "unknown_error",
                "message": f"Erro inesperado: {str(e)}"
            }

    def get_debug_status(self) -> Dict[str, Any]:
        """Obtém status debug da API incluindo modelos carregados"""
        try:
            response = requests.get(
                f"{self.api_base_url}/debug/status",
                headers=self.headers,
                timeout=10
            )
            response.raise_for_status()
            return {"success": True, "data": response.json()}
        except requests.exceptions.RequestException as e:
            return {
                "success": False,
                "message": f"Erro ao obter status: {str(e)}",
                "data": {}
            }

    def predict_baseline(self, text: str) -> Dict[str, Any]:
        """Faz predição usando o modelo baseline do MLflow Registry"""
        try:
            if not self.has_api_key():
                return {
                    "success": False,
                    "error": "API Key não configurada. Configure a variável de ambiente API_KEY ou DEEPLEARNING_API_KEY"
                }

            model_name = urllib.parse.quote("Embedding Baseline", safe='')
            response = requests.post(
                f"{self.api_base_url}/v1/predict/registry/{model_name}",
                json={"text": text},
                headers=self.headers,
                timeout=30
            )

            if response.status_code == 401:
                return {
                    "success": False,
                    "error": "API Key inválida ou não autorizada"
                }

            response.raise_for_status()
            data = response.json()
            return {
                "success": True,
                "prediction": {
                    "text": text,
                    "positive": data.get("positive", False),
                    "score": data.get("score", 0.0),
                    "sentiment": "positivo" if data.get("positive", False) else "negativo",
                    "confidence": abs(data.get("score", 0.0) - 0.5) * 2,  # Converte para 0-1
                    "request_id": data.get("request_id", ""),
                    "model": data.get("model", "baseline")
                }
            }
        except requests.exceptions.RequestException as e:
            return {
                "success": False,
                "error": f"Erro na predição baseline: {str(e)}"
            }

    def predict_production(self, text: str) -> Dict[str, Any]:
        """Faz predição usando o modelo de produção do MLflow Registry"""
        try:
            if not self.has_api_key():
                return {
                    "success": False,
                    "error": "API Key não configurada. Configure a variável de ambiente API_KEY ou DEEPLEARNING_API_KEY"
                }

            model_name = urllib.parse.quote("BiLSTM - Deep Learning", safe='')
            response = requests.post(
                f"{self.api_base_url}/v1/predict/registry/{model_name}",
                json={"text": text},
                headers=self.headers,
                timeout=30
            )

            if response.status_code == 401:
                return {
                    "success": False,
                    "error": "API Key inválida ou não autorizada"
                }

            response.raise_for_status()
            data = response.json()
            return {
                "success": True,
                "prediction": {
                    "text": text,
                    "positive": data.get("positive", False),
                    "score": data.get("score", 0.0),
                    "sentiment": "positivo" if data.get("positive", False) else "negativo",
                    "confidence": abs(data.get("score", 0.0) - 0.5) * 2,  # Converte para 0-1
                    "request_id": data.get("request_id", ""),
                    "model": data.get("model", "production")
                }
            }
        except requests.exceptions.RequestException as e:
            return {
                "success": False,
                "error": f"Erro na predição de produção: {str(e)}"
            }

    def upload_csv(self, file_content: bytes, filename: str) -> Dict[str, Any]:
        """Upload de arquivo CSV para a API"""
        try:
            if not self.has_api_key():
                return {
                    "success": False,
                    "message": "API Key não configurada. Configure a variável de ambiente API_KEY ou DEEPLEARNING_API_KEY",
                    "request_id": "",
                    "stored_local": "",
                    "stored_gcs": ""
                }

            files = {"file": (filename, file_content, "text/csv")}
            response = requests.post(
                f"{self.api_base_url}/v1/data/upload-csv",
                files=files,
                headers=self.headers,
                timeout=120  # 2 minutos para upload
            )

            if response.status_code == 401:
                return {
                    "success": False,
                    "message": "API Key inválida ou não autorizada",
                    "request_id": "",
                    "stored_local": "",
                    "stored_gcs": ""
                }

            response.raise_for_status()
            data = response.json()
            return {
                "success": True,
                "request_id": data.get("request_id", ""),
                "stored_local": data.get("stored_local", ""),
                "stored_gcs": data.get("stored_gcs", ""),
                "filename": filename,
                "message": "Upload realizado com sucesso"
            }
        except requests.exceptions.Timeout:
            return {
                "success": False,
                "message": "Timeout durante o upload. Arquivo muito grande ou conexão lenta.",
                "request_id": "",
                "stored_local": "",
                "stored_gcs": ""
            }
        except requests.exceptions.RequestException as e:
            return {
                "success": False,
                "message": f"Erro no upload: {str(e)}",
                "request_id": "",
                "stored_local": "",
                "stored_gcs": ""
            }

    def batch_predict_baseline(self, texts: List[str]) -> Dict[str, Any]:
        """Faz predições em lote usando o modelo baseline"""
        predictions = []
        failed_predictions = []

        for i, text in enumerate(texts):
            result = self.predict_baseline(text)
            if result["success"]:
                predictions.append({
                    "index": i,
                    "text": text,
                    **result["prediction"]
                })
            else:
                failed_predictions.append({
                    "index": i,
                    "text": text,
                    "error": result.get("error", "Erro desconhecido")
                })

        return {
            "success": len(predictions) > 0,
            "total_texts": len(texts),
            "successful_predictions": len(predictions),
            "failed_predictions": len(failed_predictions),
            "predictions": predictions,
            "failed": failed_predictions
        }

    def batch_predict_production(self, texts: List[str]) -> Dict[str, Any]:
        """Faz predições em lote usando o modelo de produção"""
        predictions = []
        failed_predictions = []

        for i, text in enumerate(texts):
            result = self.predict_production(text)
            if result["success"]:
                predictions.append({
                    "index": i,
                    "text": text,
                    **result["prediction"]
                })
            else:
                failed_predictions.append({
                    "index": i,
                    "text": text,
                    "error": result.get("error", "Erro desconhecido")
                })

        return {
            "success": len(predictions) > 0,
            "total_texts": len(texts),
            "successful_predictions": len(predictions),
            "failed_predictions": len(failed_predictions),
            "predictions": predictions,
            "failed": failed_predictions
        }

    # ============================
    # MLflow Metrics Endpoints
    # ============================

    def get_all_models(self) -> Dict[str, Any]:
        """Lista todos os modelos registrados no MLflow"""
        try:
            if not self.has_api_key():
                return {
                    "success": False,
                    "error": "API Key não configurada"
                }

            response = requests.get(
                f"{self.api_base_url}/v1/models",
                headers=self.headers,
                timeout=30
            )

            if response.status_code == 401:
                return {
                    "success": False,
                    "error": "API Key inválida ou não autorizada"
                }

            response.raise_for_status()
            return response.json()

        except requests.exceptions.RequestException as e:
            return {
                "success": False,
                "error": f"Erro ao buscar lista de modelos: {str(e)}"
            }

    def get_model_details(self, model_name: str) -> Dict[str, Any]:
        """Busca detalhes completos de um modelo específico (métricas, artifacts, etc)"""
        try:
            if not self.has_api_key():
                return {
                    "success": False,
                    "error": "API Key não configurada"
                }

            # URL encode do model_name para lidar com espaços e caracteres especiais
            encoded_model_name = urllib.parse.quote(model_name, safe='')

            response = requests.get(
                f"{self.api_base_url}/v1/models/{encoded_model_name}",
                headers=self.headers,
                timeout=30
            )

            if response.status_code == 401:
                return {
                    "success": False,
                    "error": "API Key inválida ou não autorizada"
                }

            response.raise_for_status()
            return response.json()

        except requests.exceptions.RequestException as e:
            return {
                "success": False,
                "error": f"Erro ao buscar detalhes do modelo '{model_name}': {str(e)}"
            }

    def get_baseline_metrics(self) -> Dict[str, Any]:
        """Busca métricas do modelo baseline via API (DEPRECATED - use get_model_details)"""
        return self.get_model_details("Embedding Baseline")

    def get_production_metrics(self) -> Dict[str, Any]:
        """Busca métricas do modelo de produção via API (DEPRECATED - use get_model_details)"""
        return self.get_model_details("BiLSTM - Deep Learning")

    def get_model_comparison(self) -> Dict[str, Any]:
        """Busca comparação entre modelos baseline e produção via API"""
        try:
            if not self.has_api_key():
                return {
                    "success": False,
                    "error": "API Key não configurada"
                }

            response = requests.get(
                f"{self.api_base_url}/v1/metrics/comparison",
                headers=self.headers,
                timeout=30
            )

            if response.status_code == 401:
                return {
                    "success": False,
                    "error": "API Key inválida ou não autorizada"
                }

            response.raise_for_status()
            return response.json()

        except requests.exceptions.RequestException as e:
            return {
                "success": False,
                "error": f"Erro ao buscar comparação de modelos: {str(e)}"
            }

    def check_mlflow_health(self) -> Dict[str, Any]:
        """Verifica se o MLflow está acessível via API"""
        try:
            if not self.has_api_key():
                return {
                    "success": False,
                    "status": "api_key_missing",
                    "message": "API Key não configurada"
                }

            response = requests.get(
                f"{self.api_base_url}/v1/mlflow/health",
                headers=self.headers,
                timeout=10
            )

            if response.status_code == 401:
                return {
                    "success": False,
                    "status": "unauthorized",
                    "message": "API Key inválida ou não autorizada"
                }

            response.raise_for_status()
            return response.json()

        except requests.exceptions.Timeout:
            return {
                "success": False,
                "status": "timeout",
                "message": "Timeout ao verificar MLflow"
            }
        except requests.exceptions.ConnectionError:
            return {
                "success": False,
                "status": "connection_error",
                "message": "Erro de conexão ao verificar MLflow"
            }
        except requests.exceptions.RequestException as e:
            return {
                "success": False,
                "status": "error",
                "message": f"Erro ao verificar MLflow: {str(e)}"
            }

