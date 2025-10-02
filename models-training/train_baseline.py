# Imports & GPU setup
import numpy as np
import pandas as pd
import tensorflow as tf
import os
import sys
import io
import json
from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    f1_score,
    accuracy_score,
    precision_score,
    recall_score
)
from tensorflow.keras import layers, models, callbacks, optimizers
from keras.saving import register_keras_serializable
from tensorflow.keras import mixed_precision

from baseline_model import build_dnn_pool
import mlflow
import mlflow.tensorflow
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, TensorSpec

# Fix encoding no Windows
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Configure Google Cloud credentials for GCS artifact storage
credentials_env = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

# Enable artifact uploads through MLflow proxy (required for remote tracking server)
os.environ["MLFLOW_ENABLE_PROXY_MULTIPART_UPLOAD"] = "true"
os.environ["MLFLOW_MULTIPART_UPLOAD_CHUNK_SIZE"] = str(100 * 1024 * 1024)  # 100MB chunks

print("GPUs:", tf.config.list_physical_devices('GPU'))

# Mixed precision só se houver GPU
if tf.config.list_physical_devices('GPU'):
    mixed_precision.set_global_policy('mixed_float16')

def train_baseline_model(data_path="datasets/Reviews.csv"):
    """
    Treina o modelo baseline (DNN Pool) com tracking MLflow
    """
    # Configure MLflow
    mlflow.set_tracking_uri("https://mlflow-server-273169854208.us-central1.run.app")

    # Set experiment (will use the default artifact root configured on server)
    # Using _gcs suffix to create new experiment with GCS artifact storage
    experiment = mlflow.set_experiment("reviews_baseline_train")
    print(f"📊 Experiment ID: {experiment.experiment_id}")
    print(f"📊 Artifact Location: {experiment.artifact_location}")

    # If artifact location is /tmp (old experiment), warn the user
    if experiment.artifact_location.startswith('/tmp'):
        print("⚠️  AVISO: Experiment configurado com artifact location local (/tmp)")
        print("⚠️  Artifacts não serão persistidos! Considere criar novo experiment.")

    # Start MLflow run
    with mlflow.start_run(run_name="baseline_dnn_pool") as run:
        print(f"🔗 Run ID: {run.info.run_id}")
        print(f"🔗 Artifact URI: {run.info.artifact_uri}")
        # 1) Dados (pré-processo)
        print(f"📥 Carregando dados de: {data_path}")
        df = pd.read_csv(data_path)
        print(f"✅ Dataset carregado: {df.shape}")
        df.columns = [c.strip() for c in df.columns]
        df["Score"] = pd.to_numeric(df["Score"], errors="coerce")

        def map_label(score):
            if pd.isna(score): return -1
            s = int(score)
            if s <= 2: return 0
            if s >= 4: return 1
            return -1

        df["label"] = df["Score"].apply(map_label).astype(int)
        df = df[(df["label"] >= 0) & df["Text"].notna() & (df["Text"].str.strip() != "")]
        df["text_raw"] = df["Text"].astype(str)

        # Split inicial
        X_train, X_test, y_train, y_test = train_test_split(
            df["text_raw"].values,
            df["label"].values.astype(int),
            test_size=0.20,
            stratify=df["label"].values,
            random_state=42
        )

        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.20, stratify=y_train, random_state=42
        )

        print("sizes (antes balanceamento):", len(X_train), len(X_val), len(X_test))

        # 1.1 Balanceamento (undersampling no treino)
        train_df = pd.DataFrame({"text": X_train, "label": y_train})
        n_min = train_df["label"].value_counts().min()
        balanced_train = (
            train_df.groupby("label", group_keys=False)
                    .sample(n=n_min, random_state=42)
                    .sample(frac=1.0, random_state=42)
                    .reset_index(drop=True)
        )
        X_train = balanced_train["text"].values
        y_train = balanced_train["label"].values

        # 2) Vetorização (Keras)
        VOCAB_SIZE = 30_000
        tok_lens = np.array([len(s.split()) for s in X_train])
        MAX_LEN = int(np.clip(np.percentile(tok_lens, 95), 100, 300))
        print("MAX_LEN =", MAX_LEN)

        @register_keras_serializable(package="preproc")
        def custom_standardize(x):
            x = tf.strings.lower(x)
            x = tf.strings.regex_replace(x, r"[^\w\s']", " ")
            x = tf.strings.regex_replace(x, r"\s+", " ")
            return tf.strings.strip(x)

        vectorizer = layers.TextVectorization(
            max_tokens=VOCAB_SIZE,
            output_mode="int",
            output_sequence_length=MAX_LEN,
            standardize=custom_standardize,
            split="whitespace"
        )
        vectorizer.adapt(tf.data.Dataset.from_tensor_slices(X_train).batch(1024))

        # 3) tf.data Datasets
        BATCH = 256
        AUTOTUNE = tf.data.AUTOTUNE

        def make_ds(X, y, training=False):
            ds = tf.data.Dataset.from_tensor_slices((X, y))
            if training:
                ds = ds.shuffle(4096, seed=42)
            return ds.batch(BATCH).prefetch(AUTOTUNE)

        train_ds = make_ds(X_train, y_train, training=True)
        val_ds   = make_ds(X_val,   y_val)
        test_ds  = make_ds(X_test,  y_test)

        # 4) Modelo
        EMBED_DIM = 128

        # Enable autologging for metrics only (disable params, models, and datasets)
        # mlflow.tensorflow.autolog(
        #     log_models=False,
        #     log_datasets=False,
        #     log_input_examples=False,
        #     log_model_signatures=False,
        #     disable=False,
        #     exclusive=False,
        #     disable_for_unsupported_versions=False,
        #     silent=False
        # )

        # Log hyperparameters manually
        mlflow.log_param("model_type", "baseline_dnn_pool")
        mlflow.log_param("vocab_size", VOCAB_SIZE)
        mlflow.log_param("embed_dim", EMBED_DIM)
        mlflow.log_param("max_len", MAX_LEN)
        mlflow.log_param("batch_size", BATCH)
        mlflow.log_param("optimizer", "Adam")
        mlflow.log_param("learning_rate", 1e-3)
        mlflow.log_param("dropout_rate", 0.3)

        model = build_dnn_pool(vectorizer, VOCAB_SIZE, EMBED_DIM)
        model.summary()

        mlflow.log_param("total_params", model.count_params())

        # 5) Treino
        ES = callbacks.EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True)
        RLROP = callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1)
        CKPT = callbacks.ModelCheckpoint("baseline_best.keras", monitor="val_loss", save_best_only=True)

        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=50,
            callbacks=[ES, RLROP, CKPT],
            verbose=1
        )

        # EarlyStopping com restore_best_weights=True já restaurou os melhores pesos
        # O modelo agora está no estado da melhor época
        best_epoch = np.argmin(history.history['val_loss'])
        print(f"\n📊 Melhor época: {best_epoch + 1}/{len(history.history['loss'])}")
        print(f"   Train Loss: {history.history['loss'][best_epoch]:.4f} | Val Loss: {history.history['val_loss'][best_epoch]:.4f}")
        print(f"   Train AUC: {history.history['auc'][best_epoch]:.4f} | Val AUC: {history.history['val_auc'][best_epoch]:.4f}")
        print(f"   Train Acc: {history.history['accuracy'][best_epoch]:.4f} | Val Acc: {history.history['val_accuracy'][best_epoch]:.4f}")

        # Log best epoch number and validation metrics from history
        mlflow.log_metric("best_epoch", best_epoch + 1)
        mlflow.log_metric("best_val_loss", history.history['val_loss'][best_epoch])
        mlflow.log_metric("best_val_accuracy", history.history['val_accuracy'][best_epoch])
        mlflow.log_metric("best_val_auc", history.history['val_auc'][best_epoch])
        mlflow.log_metric("best_val_precision", history.history['val_precision'][best_epoch])
        mlflow.log_metric("best_val_recall", history.history['val_recall'][best_epoch])

        # Salvar contagem total de épocas treinadas (pode ser útil para análise)
        mlflow.log_metric("total_epochs_trained", len(history.history['loss']))

        # 6) Threshold ótimo (val)
        y_val_proba = model.predict(val_ds, verbose=0).ravel()
        thr_grid = np.linspace(0.2, 0.8, 31)
        best_t, best_f1 = max(((t, f1_score(y_val, (y_val_proba >= t).astype(int))) for t in thr_grid),
                              key=lambda x: x[1])
        print(f"Best threshold (val): {best_t:.3f} | F1: {best_f1:.3f}")

        mlflow.log_metric("best_threshold", best_t)
        mlflow.log_metric("best_f1_val", best_f1)

        # 7) Avaliação (teste)
        y_test_proba = model.predict(test_ds, verbose=0).ravel()
        y_test_pred  = (y_test_proba >= best_t).astype(int)

        print("\n=== Baseline DNN Pool — Test ===")
        print(classification_report(y_test, y_test_pred, digits=3))
        print("Confusion matrix:\n", confusion_matrix(y_test, y_test_pred))
        roc_auc = roc_auc_score(y_test, y_test_proba)
        print("ROC-AUC:", round(roc_auc, 4))

        # Calculate additional test metrics
        test_accuracy = accuracy_score(y_test, y_test_pred)
        test_precision = precision_score(y_test, y_test_pred)
        test_recall = recall_score(y_test, y_test_pred)
        test_f1 = f1_score(y_test, y_test_pred)

        # Log all test metrics
        mlflow.log_metric("test_accuracy", test_accuracy)
        mlflow.log_metric("test_precision", test_precision)
        mlflow.log_metric("test_recall", test_recall)
        mlflow.log_metric("test_f1", test_f1)
        mlflow.log_metric("test_roc_auc", roc_auc)

        # Log confusion matrix values
        cm = confusion_matrix(y_test, y_test_pred)
        mlflow.log_metric("test_tn", int(cm[0][0]))
        mlflow.log_metric("test_fp", int(cm[0][1]))
        mlflow.log_metric("test_fn", int(cm[1][0]))
        mlflow.log_metric("test_tp", int(cm[1][1]))

        # Log dataset sizes
        mlflow.log_metric("train_size", len(X_train))
        mlflow.log_metric("val_size", len(X_val))
        mlflow.log_metric("test_size", len(X_test))

        # 8) Salvar modelo
        save_dir = "models"
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "baseline_dnn_pool.keras")
        model.save(save_path)
        print("Modelo salvo em:", save_path)

        # 9) Teste de inferência e criação de input example
        texts = ["absolutely loved this product!", "arrived broken and support ignored me"]
        proba = model.predict(tf.constant(texts, dtype=tf.string), verbose=0)
        print("probas:", proba.ravel())

        # Create TensorSpec-based signature for MLflow
        input_schema = Schema([TensorSpec(np.dtype(str), (-1,), name="text")])
        output_schema = Schema([TensorSpec(np.dtype(np.float32), (-1, 1), name="predictions")])
        signature = ModelSignature(inputs=input_schema, outputs=output_schema)

        # Log model to MLflow with signature
        mlflow.tensorflow.log_model(
            model=model,
            artifact_path="model",
            signature=signature
        )

        # Log checkpoint
        mlflow.log_artifact(save_path)

        # Log confusion matrix as text artifact
        cm_path = os.path.join(save_dir, "confusion_matrix.txt")
        with open(cm_path, "w") as f:
            f.write("Confusion Matrix (Test Set):\n")
            f.write(str(confusion_matrix(y_test, y_test_pred)))
            f.write("\n\nClassification Report:\n")
            f.write(classification_report(y_test, y_test_pred, digits=3))
        mlflow.log_artifact(cm_path)

        # Save training history as JSON artifact
        history_dict = {
            "loss": [{"epoch": i, "value": float(v)} for i, v in enumerate(history.history['loss'])],
            "val_loss": [{"epoch": i, "value": float(v)} for i, v in enumerate(history.history['val_loss'])],
            "accuracy": [{"epoch": i, "value": float(v)} for i, v in enumerate(history.history['accuracy'])],
            "val_accuracy": [{"epoch": i, "value": float(v)} for i, v in enumerate(history.history['val_accuracy'])],
            "auc": [{"epoch": i, "value": float(v)} for i, v in enumerate(history.history['auc'])],
            "val_auc": [{"epoch": i, "value": float(v)} for i, v in enumerate(history.history['val_auc'])],
            "precision": [{"epoch": i, "value": float(v)} for i, v in enumerate(history.history['precision'])],
            "val_precision": [{"epoch": i, "value": float(v)} for i, v in enumerate(history.history['val_precision'])],
            "recall": [{"epoch": i, "value": float(v)} for i, v in enumerate(history.history['recall'])],
            "val_recall": [{"epoch": i, "value": float(v)} for i, v in enumerate(history.history['val_recall'])],
        }

        history_json_path = os.path.join(save_dir, "training_history.json")
        with open(history_json_path, "w") as f:
            json.dump(history_dict, f, indent=2)
        mlflow.log_artifact(history_json_path)

        # Save model architecture summary
        model_summary_path = os.path.join(save_dir, "model_summary.txt")
        with open(model_summary_path, "w", encoding="utf-8") as f:
            model.summary(print_fn=lambda x: f.write(x + '\n'))
        mlflow.log_artifact(model_summary_path)

        # Log training history plots
        best_epoch = np.argmin(history.history['val_loss'])

        # Plot loss
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Val Loss')
        plt.axvline(x=best_epoch, color='r', linestyle='--', alpha=0.5, label=f'Best Epoch ({best_epoch+1})')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Loss over Epochs')
        plt.grid(True)

        # Plot AUC
        plt.subplot(1, 3, 2)
        plt.plot(history.history['auc'], label='Train AUC')
        plt.plot(history.history['val_auc'], label='Val AUC')
        plt.axvline(x=best_epoch, color='r', linestyle='--', alpha=0.5, label=f'Best Epoch ({best_epoch+1})')
        plt.xlabel('Epoch')
        plt.ylabel('AUC')
        plt.legend()
        plt.title('AUC over Epochs')
        plt.grid(True)

        # Plot Accuracy
        plt.subplot(1, 3, 3)
        plt.plot(history.history['accuracy'], label='Train Accuracy')
        plt.plot(history.history['val_accuracy'], label='Val Accuracy')
        plt.axvline(x=best_epoch, color='r', linestyle='--', alpha=0.5, label=f'Best Epoch ({best_epoch+1})')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.title('Accuracy over Epochs')
        plt.grid(True)

        plt.tight_layout()

        # Log as figure (viewable in MLflow UI metrics tab)
        mlflow.log_figure(plt.gcf(), "training_history.png")

        # Also save locally and log as artifact
        plot_path = os.path.join(save_dir, "training_history.png")
        plt.savefig(plot_path, dpi=100)
        plt.close()
        mlflow.log_artifact(plot_path)

        print(f"\n📊 Gráficos de treino salvos em: {plot_path}")

        # Print MLflow run info
        run = mlflow.active_run()
        print(f"\n🔗 MLflow Run ID: {run.info.run_id}")
        print(f"🔗 MLflow Run URL: {mlflow.get_tracking_uri()}/experiments/{run.info.experiment_id}/runs/{run.info.run_id}")

        return model, history, best_t

if __name__ == "__main__":
    model, history, threshold = train_baseline_model()