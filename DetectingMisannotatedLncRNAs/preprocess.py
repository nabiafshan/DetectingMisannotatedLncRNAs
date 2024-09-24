import argparse
import itertools
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import pickle
import pathlib
import sys
sys.path.append('.')

from src.data.data_utils import get_tokenizer


# truncate sequences longer than max_len
MAX_LEN = 4000
KMER = 3 # or 6

# read_dir = pathlib.Path('.../misannotated_lncrna/data/external/cppred')
# write_dir = pathlib.Path('.../misannotated_lncrna/data/interim')

def tokenize_sequences(sequences, kmer=KMER, max_len=MAX_LEN):
    """
    takes input sequences e.g. "AAAAAC" and returns tokenized representation.
    
    i've implemented a sliding window kmer tokenization, i.e. kmer window shifts 
    by 1 at each step. Therefore, 3-mer tokens of AAAAAC would be 
    [AAA, AAA, AAA, AAC] 

    (at some point, i'd like to compare to the discrete tokenization as well, 
    i.e the one where 3-mer tokens of AAAAAC would be [AAA, AAC] only)

    e.g. if tokernizer is {AAA:1, AAC: 2, AAG:3, ...}, then, 
    "AAAAAC" -> [AAA, AAA, AAA, AAC] -> [1,1,1,2]

    Most of this function has been adapted from 
    https://github.com/hzy95/EPIVAN/blob/master/sequence_processing.py

    """
 
    def sentence2word(str_set, k_mer=kmer):
        word_seq=[]
        for sr in str_set:
            tmp=[]
            for i in range(len(sr)-(k_mer-1)):
                tmp.append(sr[i:i+k_mer])
            word_seq.append(' '.join(tmp))
        return word_seq

    def word2num(wordseq, tokenizer, max_len):
        sequences = tokenizer.texts_to_sequences(wordseq)
        numseq = pad_sequences(sequences, maxlen=max_len)
        return numseq

    def sentence2num(str_set, tokenizer, max_len):
        wordseq = sentence2word(str_set)
        numseq = word2num(wordseq, tokenizer, max_len)
        return numseq

    tokenizer, _ = get_tokenizer(kmer)
    tokenized_sequences = sentence2num(sequences, tokenizer, max_len)
    return tokenized_sequences


def main(read_dir, write_dir):
    # test- coding rnas
    print('Preprocessing test coding RNAs...')
    crna_test = open(read_dir / 'Human_coding_RNA_test.fa','r').read().splitlines()
    crna_test_seqs = crna_test[1::2]
    crna_test_ids = [s.split('|')[3] for s in crna_test[0::2]]

    crna_test_tokens = tokenize_sequences(crna_test_seqs, KMER, MAX_LEN)

    with open(write_dir / 'X_c_test.pickle', 'wb') as handle:
        pickle.dump(crna_test_tokens, handle)
    with open(write_dir / 'X_c_ids_test.pickle', 'wb') as handle:
        pickle.dump(crna_test_ids, handle)

    # test- noncoding rnas
    print('Preprocessing test non-coding RNAs...')
    ncrna_test = open(read_dir / 'Homo38_ncrna_test.fa','r').read().splitlines()
    ncrna_test_seqs = ncrna_test[1::2]
    lnc_seqs_inds =  [i for i,seq in enumerate(ncrna_test_seqs) if len(seq)>200]
    ncrna_test_seqs_lnc = [ncrna_test_seqs[i] for i in lnc_seqs_inds]
    ncrna_test_ids = [s.split(' ')[0][1:] for s in ncrna_test[0::2]]
    ncrna_test_ids_lnc = [ncrna_test_ids[i] for i in lnc_seqs_inds]

    ncrna_test_tokens = tokenize_sequences(ncrna_test_seqs_lnc, KMER, MAX_LEN)

    with open(write_dir / 'X_nc_test.pickle', 'wb') as handle:
        pickle.dump(ncrna_test_tokens, handle)
    with open(write_dir / 'X_nc_ids_test.pickle', 'wb') as handle:
        pickle.dump(ncrna_test_ids_lnc, handle)

    # train- coding rnas
    print('Preprocessing train coding RNAs...')
    crna_train = open(read_dir / 'Human.coding_RNA_training.fa','r').read().splitlines()
    crna_train_seqs = crna_train[1::2]
    crna_train_ids = [s.split('|')[3] for s in crna_train[0::2]]

    crna_train_tokens = tokenize_sequences(crna_train_seqs, KMER, MAX_LEN)

    with open(write_dir / 'X_c_train.pickle', 'wb') as handle:
        pickle.dump(crna_train_tokens, handle)
    with open(write_dir / 'X_c_ids_train.pickle', 'wb') as handle:
        pickle.dump(crna_train_ids, handle)


    # train- noncoding rnas
    print('Preprocessing train coding RNAs...')
    ncrna_train = open(read_dir / 'Homo38.ncrna_training.fa','r').read().splitlines()
    ncrna_train_seqs = ncrna_train[1::2]
    lnc_seqs_inds =  [i for i,seq in enumerate(ncrna_train_seqs) if len(seq)>200]
    ncrna_train_seqs_lnc = [ncrna_train_seqs[i] for i in lnc_seqs_inds]
    ncrna_train_ids = [s.split(' ')[0][1:] for s in ncrna_train[0::2]]
    ncrna_train_ids_lnc = [ncrna_train_ids[i] for i in lnc_seqs_inds]

    ncrna_train_tokens = tokenize_sequences(ncrna_train_seqs_lnc, KMER, MAX_LEN)

    with open(write_dir / 'X_nc_train.pickle', 'wb') as handle:
        pickle.dump(ncrna_train_tokens, handle)
    with open(write_dir / 'X_nc_ids_train.pickle', 'wb') as handle:
        pickle.dump(ncrna_train_ids_lnc, handle)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--read_dir",
        help="Path to data to read in",
        required=True,
    )
    parser.add_argument(
        "-o",
        "--write_dir",
        help="Path to where new data should be written",
        required=True,
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(
        read_dir=pathlib.Path(args.read_dir),
        write_dir=pathlib.Path(args.write_dir),
    )