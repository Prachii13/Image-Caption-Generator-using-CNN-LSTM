import numpy as np
from keras.models import Model
from keras.layers import Input, Dense, LSTM, Embedding, Dropout, add
from keras.utils import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.callbacks import ModelCheckpoint
from pickle import load

# Create tokenizer from captions
def create_tokenizer(descriptions):
    lines = []
    for key in descriptions:
        lines.extend(descriptions[key])
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer

# Create input-output sequences
def create_sequences(tokenizer, max_length, descriptions, photos, vocab_size):
    X1, X2, y = [], [], []
    for key, desc_list in descriptions.items():
        for desc in desc_list:
            seq = tokenizer.texts_to_sequences([desc])[0]
            for i in range(1, len(seq)):
                in_seq, out_seq = seq[:i], seq[i]
                in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
                out_seq = np.zeros(vocab_size)
                out_seq[out_seq] = 1.0
                X1.append(photos[key][0])
                X2.append(in_seq)
                y.append(out_seq)
    return np.array(X1), np.array(X2), np.array(y)

# Max caption length
def max_length(descriptions):
    return max(len(d.split()) for desc_list in descriptions.values() for d in desc_list)

# Define model
def define_model(vocab_size, max_length):
    # Feature extractor (from CNN)
    inputs1 = Input(shape=(2048,))
    fe1 = Dropout(0.5)(inputs1)
    fe2 = Dense(25
