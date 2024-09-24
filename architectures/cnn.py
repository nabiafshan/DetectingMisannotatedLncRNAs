import tensorflow as tf
from tensorflow import keras
from keras.layers import *
from keras.models import *
from keras import optimizers
warnings.filterwarnings('ignore')

# kmer = '3mer'
# MAX_SEQUENCE_LENGTH = 4000
# EMBEDDING_DIM = 100

def get_cnn_model(embedding_matrix= embedding_matrix, kmer=kmer, MAX_SEQUENCE_LENGTH = MAX_SEQUENCE_LENGTH, EMBEDDING_DIM = EMBEDDING_DIM):  
    embedding_layer = Embedding(len(word_index),
                                EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=True,
                                mask_zero=True)
    sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH, ), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)
    x = Conv1D(64, 5, activation="relu", padding='same')(embedded_sequences)
    x = MaxPooling1D(5)(x)
    x = Conv1D(64, 5, activation="relu", padding='same')(x)
    x = MaxPooling1D(5)(x)
    x = Conv1D(64, 5, activation="relu", padding='same')(x)
    x = GlobalMaxPooling1D()(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.5)(x)
    preds = Dense(2, activation='softmax')(x)
    model = Model(sequence_input, preds)
    return model

