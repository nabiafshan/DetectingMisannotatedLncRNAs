import itertools
import numpy as np
from keras.preprocessing.text import Tokenizer
import pickle  
import pathlib

def get_tokenizer(kmerLen = 3):
    '''
    Returns tokenizer, word index (kmer length words w/ associated int indices)
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

    return tokenizer, word_index
    

def readEmbeddingMatrix(kmer: int, dataDir: pathlib.Path):
    '''
    Read embedding matrix for relevant kmer
    '''
    if kmer == 6:
        with open(dataDir / 'embedding_matrix_6mer.pickle', 'rb') as handle:
            embedding_matrix = pickle.load(handle)
    elif kmer == 3:
        with open(dataDir / 'embedding_matrix_3mer.pickle', 'rb') as handle:
            embedding_matrix = pickle.load(handle)
    return embedding_matrix


def get_train_valid_splits(read_dir: pathlib.Path):
    '''
    Reads saved pickles for tokenized train cRNA & ncRNA,
    Returns train-validation split data
    '''

    VALIDATION_SPLIT = 0.2
    RANDOM_SEED = 0

    with open(read_dir / 'X_c_train.pickle', 'rb') as handle:
        X_c = pickle.load(handle)
    with open(read_dir / 'X_nc_train.pickle', 'rb') as handle:
        X_nc = pickle.load(handle)
    data = np.vstack((X_c, X_nc))

    with open(read_dir / 'X_c_ids_train.pickle', 'rb') as handle:
        X_c_ids = pickle.load(handle)
    with open(read_dir / 'X_nc_ids_train.pickle', 'rb') as handle:
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
    np.random.seed(RANDOM_SEED)
    np.random.shuffle(indices)
    data = data[indices]
    labels = labels[indices]
    ids = np.array(ids)[indices]
    nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

    x_train = data[:-nb_validation_samples]
    y_train = labels[:-nb_validation_samples]
    ids_train = ids[:-nb_validation_samples]
    x_val = data[-nb_validation_samples:]
    y_val = labels[-nb_validation_samples:]
    ids_val = ids[-nb_validation_samples:]

    return x_train, y_train, ids_train, x_val, y_val, ids_val

