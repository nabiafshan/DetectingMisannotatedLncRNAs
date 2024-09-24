import itertools
import numpy as np
from keras.preprocessing.text import Tokenizer
import pickle  
import pathlib
import argparse
import sys
sys.path.append('.')

from src.data.data_utils import get_tokenizer 
 

# read_dir = pathlib.Path('/data/external/dna2vec')
# write_dir = pathlib.Path('/data/interim')


def getKmerEmbeddingMatrix(read_dir: pathlib.Path, kmer: int):
    """
    read_dir: path at which w2v file for dna embeddings is located
        dna2vec representations are obtained from https://arxiv.org/abs/1701.06279
    kmer: can be either 3 or 6, length of kmer for which to get embedding
    """  
    embedding_dim = 100
    # Read all embeddings
    embeddings_index = {}
    f = open(read_dir / 'dna2vec-20161219-0153-k3to8-100d-10c-29320Mbp-sliding-Xat.w2v')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    # Get word indices for relevant kmer
    _, word_index = get_tokenizer(kmer)

    # get embedding matrix
    embedding_matrix = np.zeros((len(word_index), embedding_dim))
    for word, i in word_index.items(): 
        embedding_vector = embeddings_index.get(word.upper())
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
    return embedding_matrix

def main(read_dir, write_dir):
    embedding_matrix_6mer = getKmerEmbeddingMatrix(read_dir, 6)
    embedding_matrix_3mer = getKmerEmbeddingMatrix(read_dir, 3)

    with open(write_dir / 'embedding_matrix_6mer.pickle', 'wb') as handle:
        pickle.dump(embedding_matrix_6mer, handle)
    with open(write_dir / 'embedding_matrix_3mer.pickle', 'wb') as handle:
        pickle.dump(embedding_matrix_3mer, handle)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--read_dir",
        help="Path to dna2vec file",
        required=True,
    )
    parser.add_argument(
        "-o",
        "--write_dir",
        help="Path to where kmer embeddings should be written",
        required=True,
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(
        read_dir=pathlib.Path(args.read_dir),
        write_dir=pathlib.Path(args.write_dir),
    )