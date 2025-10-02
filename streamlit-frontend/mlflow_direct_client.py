"""
Cliente MLflow direto para carregar e usar modelos sem passar pela API
"""
import os
import mlflow
import mlflow.tensorflow
import tensorflow as tf
from typing import Dict, Any
from pathlib import Path

# Configure Google Cloud credentials for GCS artifact storage
credentials_path = Path(__file__).parent / "mlflow-client-credentials.json"
if credentials_path.exists():
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(credentials_path)
    print(f"✅ Google Cloud credentials configured: {credentials_path}")
else:
    print(f"⚠️ Credentials file not found: {credentials_path}")

# Configure MLflow
MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI', "https://mlflow-server-273169854208.us-central1.run.app")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


class MLflowDirectClient:
    """Cliente para carregar modelos diretamente do MLflow Registry"""

    def __init__(self):
        self.models_cache = {}

    def load_model(self, model_name: str, force_reload: bool = False):
        """
        Carrega um modelo do MLflow Registry

        Args:
            model_name: Nome do modelo registrado (ex: "BiLSTM - Deep Learning")
            force_reload: Se True, recarrega o modelo mesmo se estiver em cache

        Returns:
            Modelo carregado ou None se falhar
        """
        if model_name in self.models_cache and not force_reload:
            return self.models_cache[model_name]

        try:
            # Carrega a versão mais recente do modelo
            client = mlflow.tracking.MlflowClient()
            print(f"🔍 Buscando versões do modelo '{model_name}'...")
            latest_version = client.get_latest_versions(model_name, stages=["None", "Staging", "Production"])[0]
            model_uri = f"models:/{model_name}/{latest_version.version}"

            print(f"🔄 Carregando modelo '{model_name}' versão {latest_version.version}...")
            print(f"📍 URI: {model_uri}")
            model = mlflow.tensorflow.load_model(model_uri)

            # Salva em cache
            self.models_cache[model_name] = model
            print(f"✅ Modelo '{model_name}' carregado com sucesso!")

            return model

        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"❌ Erro ao carregar modelo '{model_name}':")
            print(f"Tipo: {type(e).__name__}")
            print(f"Mensagem: {str(e)}")
            print(f"Traceback completo:\n{error_details}")
            return None

    def predict_baseline(self, text: str) -> Dict[str, Any]:
        """Faz predição usando o modelo Embedding Baseline"""
        return self._predict_with_model("Embedding Baseline", text)

    def predict_production(self, text: str) -> Dict[str, Any]:
        """Faz predição usando o modelo BiLSTM - Deep Learning"""
        return self._predict_with_model("BiLSTM - Deep Learning", text)

    def _predict_with_model(self, model_name: str, text: str) -> Dict[str, Any]:
        """
        Faz predição com um modelo específico

        Args:
            model_name: Nome do modelo no registry
            text: Texto para classificar

        Returns:
            Dict com resultado da predição
        """
        try:
            # Carrega o modelo (usa cache se disponível)
            model = self.load_model(model_name)

            if model is None:
                return {
                    "success": False,
                    "error": f"Modelo '{model_name}' não pôde ser carregado. Verifique os logs no terminal para mais detalhes."
                }

            # Faz predição
            prediction = model.predict(tf.constant([text], dtype=tf.string), verbose=0)
            score = float(prediction[0][0])
            positive = score >= 0.5

            return {
                "success": True,
                "prediction": {
                    "text": text,
                    "positive": positive,
                    "score": score,
                    "sentiment": "positivo" if positive else "negativo",
                    "confidence": abs(score - 0.5) * 2,  # Converte para 0-1
                    "model": model_name
                }
            }

        except Exception as e:
            return {
                "success": False,
                "error": f"Erro na predição com '{model_name}': {str(e)}"
            }
