from keras.layers import *
from keras.models import *

from transformer import TokenAndPositionEmbedding, TransformerBlock


def misannot_CNN_model(
    embedding_matrix,
    len_word_index=65,
    embedding_dim=100,
    max_sequence_len=4000,
):
    """
    Creates CNN model used for the identifying misannotated lncrnas
    """
    embedding_layer = Embedding(
        len_word_index,
        embedding_dim,
        weights=[embedding_matrix],
        input_length=max_sequence_len,
        trainable=True,
        mask_zero=True,
    )
    sequence_input = Input(shape=(max_sequence_len,), dtype="int32")
    embedded_sequences = embedding_layer(sequence_input)
    x = Conv1D(64, 7, activation="relu", padding="same")(embedded_sequences)
    x = MaxPooling1D(5)(x)
    x = Conv1D(64, 7, activation="relu", padding="same")(x)
    x = MaxPooling1D(5)(x)
    x = Conv1D(64, 7, activation="relu", padding="same")(x)
    x = GlobalMaxPooling1D()(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.5)(x)
    preds = Dense(2, activation="softmax")(x)
    model = Model(sequence_input, preds)
    return model


def misannot_LSTM_model(
    embedding_matrix,
    len_word_index=65,
    embedding_dim=100,
    max_sequence_len=4000,
):
    """
    Creates LSTM model used for the identifying misannotated lncrnas
    """
    embedding_layer = Embedding(
        len_word_index,
        embedding_dim,
        weights=[embedding_matrix],
        input_length=max_sequence_len,
        trainable=True,
        mask_zero=True,
    )
    sequence_input = Input(shape=(max_sequence_len,), dtype="int32")
    embedded_sequences = embedding_layer(sequence_input)
    x = Conv1D(64, 7, activation="relu", padding="same")(embedded_sequences)
    x = MaxPooling1D(5)(x)
    x = Conv1D(64, 7, activation="relu", padding="same")(x)
    x = MaxPooling1D(5)(x)
    x = Conv1D(64, 7, activation="relu", padding="same")(x)
    x = Bidirectional(LSTM(32, return_sequences=False))(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.5)(x)
    preds = Dense(2, activation="softmax")(x)
    model = Model(sequence_input, preds)
    return model


def misannot_transformer_model(
    embedding_matrix,
    len_word_index=65,
    embedding_dim=100,
    max_sequence_len=4000,
):
    """
    Creates transformer model used for the identifying misannotated lncrnas
    """
    inputs = Input(shape=(max_sequence_len,))
    embedding_layer = TokenAndPositionEmbedding(
        max_sequence_len, len_word_index, embedding_dim, embedding_matrix
    )
    x = embedding_layer(inputs)
    transformer_block = TransformerBlock(
        embed_dim=embedding_dim, num_heads=4, ff_dim=128
    )
    attn1, x = transformer_block(x)
    x = GlobalAveragePooling1D()(x)
    x = Dropout(0.5)(x)
    x = Dense(
        units=128,
        activation="relu",
    )(x)
    x = Dropout(0.5)(x)
    outputs = Dense(2, activation="softmax")(x)
    model = Model(inputs=inputs, outputs=outputs)
    return model
