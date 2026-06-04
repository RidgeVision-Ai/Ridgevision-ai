from backend.core.config import CLASS_LABELS


def build_fusion_model(image_shape: tuple[int, int, int] = (224, 224, 3), texture_dim: int = 11):
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise RuntimeError("TensorFlow is required to build the trainable fusion model.") from exc

    image_input = tf.keras.Input(shape=image_shape, name="fingerprint_image")
    texture_input = tf.keras.Input(shape=(texture_dim,), name="texture_features")

    backbone = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_tensor=image_input,
    )
    backbone.trainable = False

    cnn_features = tf.keras.layers.GlobalAveragePooling2D(name="cnn_pool")(backbone.output)
    texture_features = tf.keras.layers.Dense(64, activation="relu", name="texture_projection")(texture_input)
    fused = tf.keras.layers.Concatenate(name="fusion")([cnn_features, texture_features])

    channel_attention = tf.keras.layers.Dense(fused.shape[-1], activation="sigmoid", name="cbam_channel_gate")(fused)
    attended = tf.keras.layers.Multiply(name="attention_fusion")([fused, channel_attention])
    hidden = tf.keras.layers.Dense(256, activation="relu")(attended)
    hidden = tf.keras.layers.Dropout(0.35)(hidden)
    output = tf.keras.layers.Dense(len(CLASS_LABELS), activation="softmax", name="blood_group")(hidden)

    return tf.keras.Model(inputs=[image_input, texture_input], outputs=output, name="ridgevision_fusion_net")
