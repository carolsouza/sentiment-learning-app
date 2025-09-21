import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
import mlflow
import mlflow.sklearn
from datetime import datetime
import os
import pickle
from typing import Dict, Any, Tuple, List
import logging

logger = logging.getLogger(__name__)


class MLService:
    def __init__(self, mlflow_tracking_uri: str = "http://localhost:5000"):
        try:
            mlflow.set_tracking_uri(mlflow_tracking_uri)
            logger.info(f"MLflow tracking URI set to: {mlflow_tracking_uri}")
        except Exception as e:
            logger.warning(f"Failed to set MLflow tracking URI: {e}")
            # Fallback to local file tracking
            mlflow.set_tracking_uri("file:./mlruns")
            logger.info("Using local file tracking as fallback")

    def prepare_data(self, dataset_path: str, max_samples: int, balance_data: bool = True) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        """Prepara os dados para treinamento"""
        try:
            # Carrega o dataset
            df = pd.read_csv(dataset_path)

            # Normaliza headers para lowercase
            df.columns = df.columns.str.lower()

            # Filtra apenas scores 1, 2, 4, 5 (remove neutros)
            df_filtered = df[df["score"].isin([1, 2, 4, 5])].copy()

            # Converte para classificação binária
            df_filtered["sentiment"] = df_filtered["score"].apply(
                lambda x: "negativo" if x in [1, 2] else "positivo"
            )

            # Balanceamento se solicitado
            if balance_data:
                neg_count = len(df_filtered[df_filtered["sentiment"] == "negativo"])
                pos_count = len(df_filtered[df_filtered["sentiment"] == "positivo"])

                # Verifica se é possível balancear
                if neg_count == 0 or pos_count == 0:
                    logger.warning("Cannot balance data: only one class available")
                    raise ValueError("Não é possível balancear: apenas uma classe disponível no dataset")

                # Define o mínimo entre as classes e o limite de amostras
                samples_per_class = min(neg_count, pos_count, max_samples // 2)

                if samples_per_class < 50:  # Mínimo de 50 por classe
                    logger.warning(f"Very few samples per class: {samples_per_class}")

                # Samplea equilibradamente
                df_neg_sample = df_filtered[df_filtered["sentiment"] == "negativo"].sample(
                    n=samples_per_class, random_state=42
                )
                df_pos_sample = df_filtered[df_filtered["sentiment"] == "positivo"].sample(
                    n=samples_per_class, random_state=42
                )

                df_sample = pd.concat([df_neg_sample, df_pos_sample]).sample(
                    frac=1, random_state=42
                ).reset_index(drop=True)

                logger.info(f"Balanced sampling: {samples_per_class} per class, total: {len(df_sample)}")
            else:
                # Limita a quantidade de amostras
                if len(df_filtered) > max_samples:
                    df_sample = df_filtered.sample(n=max_samples, random_state=42).reset_index(drop=True)
                    logger.info(f"Random sampling: {max_samples} from {len(df_filtered)} available samples")
                else:
                    df_sample = df_filtered
                    logger.info(f"Using all available samples: {len(df_filtered)}")

            X = df_sample["text"].fillna("")
            y = df_sample["sentiment"]

            final_neg = len(df_sample[df_sample['sentiment'] == 'negativo'])
            final_pos = len(df_sample[df_sample['sentiment'] == 'positivo'])

            logger.info(f"Final dataset: {len(df_sample)} samples, {final_neg} negative, {final_pos} positive")

            # Verifica se há amostras suficientes
            if len(df_sample) < 100:
                logger.warning(f"Very small dataset: {len(df_sample)} samples")

            if final_neg == 0 or final_pos == 0:
                logger.error("Only one class in final dataset!")
                raise ValueError("Dataset final contém apenas uma classe. Não é possível treinar modelo.")

            return df_sample, X, y

        except Exception as e:
            logger.error(f"Error preparing data: {str(e)}")
            raise

    def train_model(
        self,
        dataset_path: str,
        max_samples: int,
        balance_data: bool = True,
        test_size: float = 0.2,
        max_features: int = 10000,
        experiment_name: str = "sentiment_analysis"
    ) -> Dict[str, Any]:
        """Treina o modelo Naive Bayes + TF-IDF"""

        # Configura o experimento
        mlflow.set_experiment(experiment_name)

        with mlflow.start_run() as run:
            try:
                # Preparação dos dados
                df_sample, X, y = self.prepare_data(dataset_path, max_samples, balance_data)

                # Log dos parâmetros
                mlflow.log_param("dataset_path", dataset_path)
                mlflow.log_param("max_samples", max_samples)
                mlflow.log_param("balance_data", balance_data)
                mlflow.log_param("test_size", test_size)
                mlflow.log_param("max_features", max_features)
                mlflow.log_param("total_samples", len(df_sample))

                # Divisão treino/teste
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=42, stratify=y
                )

                mlflow.log_param("train_samples", len(X_train))
                mlflow.log_param("test_samples", len(X_test))

                # TF-IDF Vectorization
                vectorizer = TfidfVectorizer(
                    max_features=max_features,
                    stop_words="english",
                    ngram_range=(1, 2),
                    min_df=2,
                )

                X_train_tfidf = vectorizer.fit_transform(X_train)
                X_test_tfidf = vectorizer.transform(X_test)

                # Treinamento do modelo
                model = MultinomialNB(alpha=1.0)
                model.fit(X_train_tfidf, y_train)

                # Predições
                y_pred = model.predict(X_test_tfidf)
                y_pred_proba = model.predict_proba(X_test_tfidf)

                # Calcula métricas
                accuracy = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred, average="weighted")
                precision = precision_score(y_test, y_pred, average="weighted")
                recall = recall_score(y_test, y_pred, average="weighted")

                # Log das métricas
                mlflow.log_metric("accuracy", accuracy)
                mlflow.log_metric("f1_score", f1)
                mlflow.log_metric("precision", precision)
                mlflow.log_metric("recall", recall)

                # Salva modelo e vectorizer no MLflow
                # Salva o modelo sklearn
                mlflow.sklearn.log_model(
                    model,
                    "model",
                    registered_model_name=f"sentiment_nb_{experiment_name}"
                )

                # Salva o vectorizer como artifact temporário
                import tempfile
                with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as temp_file:
                    pickle.dump(vectorizer, temp_file)
                    temp_vectorizer_path = temp_file.name

                mlflow.log_artifact(temp_vectorizer_path, "vectorizer")

                # Remove arquivo temporário
                os.unlink(temp_vectorizer_path)

                # Log da matriz de confusão
                cm = confusion_matrix(y_test, y_pred)
                mlflow.log_dict(
                    {
                        "confusion_matrix": cm.tolist(),
                        "labels": model.classes_.tolist()
                    },
                    "confusion_matrix.json"
                )

                # Log do relatório de classificação
                report = classification_report(y_test, y_pred, output_dict=True)
                mlflow.log_dict(report, "classification_report.json")

                # Model URI
                model_uri = mlflow.get_artifact_uri("model")

                logger.info(f"Training completed successfully. Run ID: {run.info.run_id}")

                # Informações sobre o balanceamento
                final_neg = len(df_sample[df_sample['sentiment'] == 'negativo'])
                final_pos = len(df_sample[df_sample['sentiment'] == 'positivo'])

                balancing_info = {
                    "was_balanced": balance_data,
                    "final_negative_count": final_neg,
                    "final_positive_count": final_pos,
                    "is_perfectly_balanced": final_neg == final_pos
                }

                return {
                    "success": True,
                    "message": "Modelo treinado com sucesso",
                    "experiment_id": run.info.experiment_id,
                    "run_id": run.info.run_id,
                    "metrics": {
                        "accuracy": accuracy,
                        "f1_score": f1,
                        "precision": precision,
                        "recall": recall
                    },
                    "confusion_matrix": cm.tolist(),
                    "class_metrics": report,
                    "balancing_info": balancing_info,
                    "model_uri": model_uri
                }

            except Exception as e:
                logger.error(f"Training failed: {str(e)}")
                mlflow.log_param("error", str(e))
                return {
                    "success": False,
                    "message": f"Erro durante o treinamento: {str(e)}",
                    "experiment_id": run.info.experiment_id,
                    "run_id": run.info.run_id,
                    "metrics": {},
                    "model_uri": ""
                }

    def predict(self, text: str, model_uri: str) -> Dict[str, Any]:
        """Faz predição usando um modelo treinado"""
        try:
            # Carrega o modelo
            model = mlflow.sklearn.load_model(model_uri)

            # Carrega o vectorizer do MLflow
            run_id = model_uri.split("/")[-2]

            # Baixa o vectorizer do MLflow artifacts
            vectorizer_path = mlflow.artifacts.download_artifacts(
                f"runs:/{run_id}/vectorizer",
                dst_path="temp_artifacts"
            )

            # O download retorna o caminho da pasta, precisa do arquivo específico
            import glob
            vectorizer_file = glob.glob(f"{vectorizer_path}/*.pkl")[0]

            with open(vectorizer_file, "rb") as f:
                vectorizer = pickle.load(f)

            # Limpa arquivos temporários
            import shutil
            if os.path.exists("temp_artifacts"):
                shutil.rmtree("temp_artifacts")

            # Transforma o texto
            text_tfidf = vectorizer.transform([text])

            # Faz a predição
            prediction = model.predict(text_tfidf)[0]
            probabilities = model.predict_proba(text_tfidf)[0]

            # Mapeia probabilidades para classes
            prob_dict = {}
            for i, class_name in enumerate(model.classes_):
                prob_dict[class_name] = float(probabilities[i])

            return {
                "prediction": prediction,
                "probabilities": prob_dict
            }

        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise

