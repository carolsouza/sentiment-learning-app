import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, regularizers


def build_bilstm64_dnn(vectorizer, VOCAB_SIZE, EMBED_DIM):
    """
    BiLSTM estável + head denso.
    return_sequences=True + GlobalMaxPooling1D para reduzir sensibilidade a ruído.
    """
    inp = layers.Input(shape=(), dtype=tf.string, name="text")
    x = vectorizer(inp)  # (B, T)
    x = layers.Embedding(VOCAB_SIZE, EMBED_DIM, name="embed", mask_zero=True)(x)

    # BiLSTM com dropout moderado; clipnorm ajuda contra exploding grads
    x = layers.Bidirectional(
        layers.LSTM(64, return_sequences=True, dropout=0.3, recurrent_dropout=0.2)
    )(x)

    # Pooling robusto (tenta também GlobalAveragePooling1D + GlobalMaxPooling1D concatenados)
    x = layers.GlobalMaxPooling1D()(x)

    # Head DNN
    x = layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.Dropout(0.4)(x)

    out = layers.Dense(1, activation="sigmoid", dtype="float32")(x)

    model = models.Model(inp, out, name="bilstm64_dnn")
    opt = optimizers.Adam(0.0005, clipnorm=1.0)  # clipnorm p/ estabilidade
    model.compile(
        loss="binary_crossentropy",
        optimizer=opt,
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.TruePositives(name="tp"),
            tf.keras.metrics.TrueNegatives(name="tn"),
            tf.keras.metrics.FalsePositives(name="fp"),
            tf.keras.metrics.FalseNegatives(name="fn")
        ]
    )
    return model