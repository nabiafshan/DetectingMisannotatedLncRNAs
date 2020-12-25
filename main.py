# Do imports from other files
from preprocess import *
from models import *
from callbacks import *
from dynamics import *

import pandas as pd
import pickle, os, sys, itertools, warnings
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras import regularizers
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, classification_report
from keras.layers import *
from keras.models import *
from keras import optimizers
warnings.filterwarnings('ignore')

# Set up directories
modelDir = ''
dataDir = ''

# Some values
maxlen = MAX_SEQUENCE_LENGTH = seq_len = 4000 
EMBEDDING_DIM=100
BATCH_SIZE = 64

# Get data
word_index = tokenizerGetWordIndex()
x_train, y_train, ids_train = getTrainData()
x_val, y_val, ids_val = getTestData()
embedding_matrix = readEmbeddingMatrix('3mer', dataDir)      
vocab_size = len(word_index) 

# Add class imbalance- penalize misclaf of cRNA more
counts = np.bincount(y_train)
weight_for_0 = 1.0 / counts[0]
weight_for_1 = 5.0 / counts[1]
class_weight = {0: weight_for_0, 1: weight_for_1}

# Get dataset
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
test_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
train_dataset = train_dataset.batch(BATCH_SIZE)
test_dataset = test_dataset.batch(BATCH_SIZE)

# Get model
model = get_cnn_model(embedding_matrix=embedding_matrix, kmer='3mer', MAX_SEQUENCE_LENGTH=MAX_SEQUENCE_LENGTH, EMBEDDING_DIM=EMBEDDING_DIM)
print(model.summary())

# Callbacks
modelCheckpointFile = os.path.join(modelDir, 'cnn.{epoch:02d}-{val_loss:.2f}.h5')
logDir = os.path.join(modelDir, 'logs_cnn')
predictions=prediction_history()

my_callbacks = [
    tf.keras.callbacks.LearningRateScheduler(scheduler),
    tf.keras.callbacks.EarlyStopping(patience=5),
    tf.keras.callbacks.ModelCheckpoint(filepath=modelCheckpointFile,
                                       save_weights_only=True,
                                       save_best_only = True),
    tf.keras.callbacks.TensorBoard(log_dir=logDir, 
                                   histogram_freq=5, 
                                   write_graph=True,
                                   write_images=True),
    predictions

]


# Compile
opt = tf.keras.optimizers.Adam(learning_rate=0.0001)
model.compile(optimizer=opt, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
history = model.fit(
    train_dataset, 
    epochs=20, 
    validation_data=test_dataset,
    callbacks=my_callbacks,
    class_weight=class_weight
)


# Make predictions
Y_pred = model.predict(x_val)
Y_pred2 = [np.argmax(y, axis=None, out=None) for y in Y_pred]
print('Confusion matrix: ')
print(confusion_matrix(y_val, np.rint(Y_pred2)))
print('Classification report: ')
print(classification_report(y_val, np.rint(Y_pred2)))


# Get prediction history
predhis_train = predictions.predhis_train
predhis_valid = predictions.predhis_valid

# Get training dynamics stats
corr_train, conf_train, var_train = get_stats(predhis_train, y_train)
corr_valid, conf_valid, var_valid = get_stats(predhis_valid, y_val)

# Save stats
df_train = pd.DataFrame({'id': ids_train,
                         'gold_lab': y_train,
                         'mean': conf_train,
                         'std': var_train})
df_val = pd.DataFrame({'id': ids_val,
                       'gold_lab': y_val,
                       'mean': conf_valid,
                       'std': var_valid})
df_both = df_train.append(df_val)
df_both.to_csv(dataDir + 'cnn_predict_hist.csv')



