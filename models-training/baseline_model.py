import tensorflow as tf
from tensorflow.keras import layers, models, optimizers


def build_dnn_pool(vectorizer, VOCAB_SIZE, EMBED_DIM):
    """
    Baseline forte: Embedding -> GlobalAveragePooling1D -> DNN.
    Rápido, estável, ótimo p/ comparar contra LSTM.
    """
    inp = layers.Input(shape=(), dtype=tf.string, name="text")
    x = vectorizer(inp)  # (B, T)
    x = layers.Embedding(VOCAB_SIZE, EMBED_DIM, name="embed", mask_zero=True)(x)
    # Pooling médio (poderia somar com max pooling se quiser)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation="relu")(x)
    out = layers.Dense(1, activation="sigmoid", dtype="float32")(x)

    model = models.Model(inp, out, name="dnn_pool")
    opt = optimizers.Adam(1e-3)
    model.compile(
        loss="binary_crossentropy",
        optimizer=opt,
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall")
        ]
    )

    return model