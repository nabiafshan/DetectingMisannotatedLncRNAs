import pickle, os, sys, itertools, warnings
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras import regularizers
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import *
from keras.models import *
from keras.models import Model
from keras import optimizers
from keras import initializers
warnings.filterwarnings('ignore')

dataDir = ''


def tokenizerGetWordIndex(kmerLen = 3):
  '''
  Returns kmer length words w/ associated indices
  '''
  
  f= ['a','c','g','t']
  res=[]

  if kmerLen == 6:
    c = itertools.product(f,f,f,f,f,f)
    for i in c:
      temp=i[0]+i[1]+i[2]+i[3]+i[4]+i[5]
      res.append(temp)
  elif kmerLen == 3:
    c = itertools.product(f,f,f)
    for i in c:
      temp=i[0]+i[1]+i[2]
      res.append(temp)
  
  res=np.array(res)
  NB_WORDS = len(res) + 1
  tokenizer = Tokenizer(num_words=NB_WORDS)
  tokenizer.fit_on_texts(res)
  word_index = tokenizer.word_index
  word_index['null']=0

  return word_index

def readEmbeddingMatrix(kmer, dataDir):
  '''
  Read embedding matrix for relevant kmer
  '''
  if kmer == '6mer':
    with open(dataDir + 'embedding_matrix_6mer.pickle', 'rb') as handle:
      embedding_matrix = pickle.load(handle)
  elif kmer == '3mer':
    with open(dataDir + 'embedding_matrix_3mer.pickle', 'rb') as handle:
      embedding_matrix = pickle.load(handle)
  return embedding_matrix




def getTrainData(data_dir = dataDir):

  '''
  Reads saved pickles for cRNA & ncRNA,
  Returns train-validation split data
  '''
  with open(data_dir +'X_c.pickle', 'rb') as handle:
      X_c = pickle.load(handle)
  with open(data_dir +'X_nc.pickle', 'rb') as handle:
      X_nc = pickle.load(handle)
  data = np.vstack((X_c, X_nc))

  with open(data_dir +'X_c_ids.pickle', 'rb') as handle:
      X_c_ids = pickle.load(handle)
  with open(data_dir +'X_nc_ids.pickle', 'rb') as handle:
      X_nc_ids = pickle.load(handle)
  ids = X_c_ids + X_nc_ids

  len_cRNA  = len(X_c)
  len_ncRNA = len(X_nc)

  # y=1: coding RNA
  # y=0: non-coding RNA
  Y = np.concatenate([ np.ones((len_cRNA), dtype=int), 
                      np.zeros((len_ncRNA), dtype=int) ])
  labels = Y
  # labels = to_categorical(np.asarray(labels))
  # print('Shape of data tensor:', data.shape)
  # print('Shape of label tensor:', labels.shape)

  # split the data into a training set and a validation set
  indices = np.arange(data.shape[0])
  np.random.seed(0)
  np.random.shuffle(indices)
  data = data[indices]
  labels = labels[indices]
  ids = np.array(ids)[indices]
  
  return data, labels, ids



  

def getTestData(data_dir = dataDir):

  '''
  Reads saved pickles for cRNA & ncRNA,
  Returns train-validation split data
  '''
  with open(data_dir +'X_c_test.pickle', 'rb') as handle:
      X_c = pickle.load(handle)
  with open(data_dir +'X_nc_test.pickle', 'rb') as handle:
      X_nc = pickle.load(handle)
  data = np.vstack((X_c, X_nc))

  with open(data_dir +'X_c_ids_test.pickle', 'rb') as handle:
      X_c_ids = pickle.load(handle)
  with open(data_dir +'X_nc_ids_test.pickle', 'rb') as handle:
      X_nc_ids = pickle.load(handle)
  ids = X_c_ids + X_nc_ids

  len_cRNA  = len(X_c)
  len_ncRNA = len(X_nc)

  # y=1: coding RNA
  # y=0: non-coding RNA
  Y = np.concatenate([ np.ones((len_cRNA), dtype=int), 
                      np.zeros((len_ncRNA), dtype=int) ])
  labels = Y
  # labels = to_categorical(np.asarray(labels))
  print('Shape of data tensor:', data.shape)
  print('Shape of label tensor:', labels.shape)

  # split the data into a training set and a validation set
  indices = np.arange(data.shape[0])
  np.random.seed(0)
  np.random.shuffle(indices)
  data = data[indices]
  labels = labels[indices]
  ids = np.array(ids)[indices]
  
  return data, labels, ids