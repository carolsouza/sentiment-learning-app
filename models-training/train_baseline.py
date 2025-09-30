# Imports & GPU setup
import numpy as np
import pandas as pd
import tensorflow as tf
import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score

from tensorflow.keras import layers, models, callbacks, optimizers
from keras.saving import register_keras_serializable
from tensorflow.keras import mixed_precision

from baseline_model import build_dnn_pool
import mlflow
import mlflow.tensorflow

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

    # Set experiment
    mlflow.set_experiment("reviews_baseline_dnn")

    # Start MLflow run
    with mlflow.start_run(run_name="baseline_dnn_pool"):
        # Enable autologging for Keras/TensorFlow
        mlflow.tensorflow.autolog()

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

        # Log hyperparameters
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

        # Log final metrics
        mlflow.log_metric("test_roc_auc", roc_auc)
        mlflow.log_metric("test_f1", f1_score(y_test, y_test_pred))

        # 8) Salvar modelo
        save_dir = "models"
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "baseline_dnn_pool.keras")
        model.save(save_path)
        print("Modelo salvo em:", save_path)

        # Log model to MLflow
        mlflow.tensorflow.log_model(model, "model")

        # 9) Teste de inferência
        texts = ["absolutely loved this product!", "arrived broken and support ignored me"]
        proba = model.predict(tf.constant(texts, dtype=tf.string), verbose=0)
        print("probas:", proba.ravel())

        return model, history, best_t

if __name__ == "__main__":
    model, history, threshold = train_baseline_model()