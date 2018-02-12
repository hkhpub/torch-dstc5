import itertools
import gensim
import re
import numpy as np
from collections import Counter

GLOVE_VEC_100D_FILE = '/hdd/data/embedding/glove-vectors/glove.6B.100d.txt'
GLOVE_VEC_300D_FILE = '/hdd/data/embedding/glove-vectors/glove.6B.300d.txt'
WORD2VEC_300D_FILE = '/hdd/data/embedding/word2vec/GoogleNews-vectors-negative300.bin'

np.random.seed(0)

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!.?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'m", " \'am", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\.", " . ", string)
    string = re.sub(r"\?", " ? ", string)
    string = re.sub(r"\(", " ( ", string)
    string = re.sub(r"\)", " ) ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

def tokenize_and_lower(_text):
    text = clean_str(_text)
    return text

def load_params(param_filenm):
    params = dict()
    with open(param_filenm, "r") as f:
        for line in f.readlines():
            kv = line.strip().split(":")
            params[kv[0].strip()] = kv[1].strip()
    return params

def build_vocab(sentences):
    """
    Builds a vocabulary mapping from word to index based on the sentences.
    Returns vocabulary mapping and inverse vocabulary mapping.
    """
    # Build vocabulary
    word_counts = Counter(itertools.chain(*sentences))

    # Mapping from index to word
    vocabulary_inv = [x[0] for x in word_counts.most_common()]
    # Mapping from word to index
    vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
    return [vocabulary, vocabulary_inv]

def pad_sentences(sentences, max_sent_len=0, padding_word = "<PAD/>"):
    """
    Pads all sentences to the same length. The length is defined by the longest sentence.
    Returns padded sentences.
    """
    if max_sent_len == 0:
        sequence_length = max(len(x) for x in sentences)
    else:
        sequence_length = max_sent_len
    padded_sentences = []
    for i in range(len(sentences)):
        sentence = sentences[i]
        num_padding = sequence_length - len(sentence)
        new_sentence = sentence + [padding_word] * num_padding
        padded_sentences.append(new_sentence)

    return padded_sentences

def build_input_data(sentences, vocabulary):
    x = np.array([[vocabulary.get(word, 0) for word in sentence] for sentence in sentences])
    return x

def load_embedding(vocabulary, embedding_dim, embedding="random"):
    embedding_matrix = np.random.uniform(low=-1.0, high=1.0, size=(len(vocabulary), embedding_dim))
    embedding_matrix = np.asarray(embedding_matrix, dtype='float32')

    if embedding == 'glove':
        glove_vec_file = GLOVE_VEC_100D_FILE if embedding_dim == 100 else GLOVE_VEC_300D_FILE
        print("loading glove embedding..")
        hit_count = 0
        with open(glove_vec_file, "r") as f:
            for line in f.readlines():
                values = line.split()
                word = values[0]
                if word in vocabulary:
                    coefs = np.asarray(values[1:], dtype='float32')
                    embedding_matrix[vocabulary[word]] = coefs
                    hit_count += 1
        print("glove miss rate: %.2f%%" % (100 - hit_count*100.0/len(vocabulary)))
        f.close()
        pass

    elif embedding == 'word2vec':
        print("loading word2vec embedding..")
        gensim_w2v = gensim.models.KeyedVectors.load_word2vec_format(WORD2VEC_300D_FILE, binary=True)
        miss_count = 0
        for word, word_id in vocabulary.items():
            if word in gensim_w2v.wv:
                embedding_matrix[word_id] = gensim_w2v[word]
            else:
                miss_count += 1
        print("word2vec miss rate: %.2f%%" % (miss_count*100.0 / len(vocabulary)))
        pass

    else:
        pass

    return embedding_matrix
    pass