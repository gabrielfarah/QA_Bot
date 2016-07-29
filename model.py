import numpy as np
np.random.seed(1337) # for reproducibility

import pandas as pd
import re

from keras.models import model_from_json

from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import TimeDistributed
from keras.layers.core import Dense, RepeatVector, Activation, Dropout
from keras.preprocessing.sequence import pad_sequences
from keras.layers.embeddings import Embedding


def tokenize(sent):
    '''Return the tokens of a sentence including punctuation.
    >>> tokenize('Bob dropped the apple. Where is the apple?')
    ['Bob', 'dropped', 'the', 'apple', '.', 'Where', 'is', 'the', 'apple', '?']
    '''
    return [x.strip() for x in re.split('(\W+)?', sent) if x.strip()]


def save_model_to_file (model, struct_file, weights_file ):
    # Save model structure
    model_struct = model.to_json ()
    open (struct_file, 'w') .write (model_struct)
    # Save model weights
    model.save_weights (weights_file, overwrite = True)

def load_model (struct_file, weights_file):
    # Load the NN structure
    model = model_from_json (open (struct_file, 'r') .read ())
    model. compile (loss = "categorical_crossentropy ", optimizer = 'sgd')
    # Restore the NN weights
    model.load_weights (weights_file)
    # Return the saved model
    return model

class SimpleSeq2seq(Sequential):
    '''
    Simple model for sequence to sequence learning.
    The encoder encodes the input sequence to vector (called context vector)
    The decoder decoder the context vector in to a sequence of vectors.
    There is no one on one relation between the input and output sequence elements.
    The input sequence and output sequence may differ in length.
    Arguments:
    output_dim : Required output dimension.
    hidden_dim : The dimension of the internal representations of the model.
    output_length : Length of the required output sequence.
    depth : Used to create a deep Seq2seq model. For example, if depth = 3, 
            there will be 3 LSTMs on the enoding side and 3 LSTMs on the 
            decoding side. You can also specify depth as a tuple. For example,
            if depth = (4, 5), 4 LSTMs will be added to the encoding side and
            5 LSTMs will be added to the decoding side.
    dropout : Dropout probability in between layers.
    '''
    def __init__(self, output_dim, hidden_dim, output_length, depth=1, dropout=0.25, **kwargs):
        super(SimpleSeq2seq, self).__init__()
        if type(depth) not in [list, tuple]:
            depth = (depth, depth)
        self.encoder = LSTM(hidden_dim, **kwargs)
        self.decoder = LSTM(hidden_dim, return_sequences=True, **kwargs)
        for i in range(1, depth[0]):
            self.add(LSTM(hidden_dim, return_sequences=True, **kwargs))
            self.add(Dropout(dropout))
        self.add(self.encoder)
        self.add(Dropout(dropout))
        self.add(RepeatVector(output_length))
        self.add(self.decoder)
        for i in range(1, depth[1]):
            self.add(LSTM(hidden_dim, return_sequences=True, **kwargs))
            self.add(Dropout(dropout))
        self.add(TimeDistributed(Dense(output_dim)))

def build_model (vocab_size, input_size, max_out_seq_len, hidden_size, input_shape=None):
    # model = Sequential()
    # model.add(LSTM (input_dim = input_size, output_dim = hidden_size, return_sequences = False))
    # model.add(Dense(hidden_size, activation = "relu"))
    # model.add(RepeatVector(max_out_seq_len))
    # model.add(LSTM(hidden_size, return_sequences = True))
    # model.add(TimeDistributed(Dense (output_dim = input_size, activation = "linear")))
    # model.compile(loss = "mean_squared_error", optimizer = 'adam')


    # embedding_size = 64
    # hidden_size = 512
    # print('Build model...')
    # model = Sequential()
    # model.add(Embedding(vocab_size, embedding_size))
    # model.add(LSTM(input_dim = input_size, output_dim = hidden_size, return_sequences = False))#LSTM(embedding_size, hidden_size)) # try using a GRU instead, for fun
    # model.add(Dense(input_dim = hidden_size, output_dim = hidden_size))
    # model.add(Activation('relu'))
    # model.add(RepeatVector(max_out_seq_len))
    # model.add(LSTM(input_dim = hidden_size, output_dim = hidden_size, return_sequences=True))
    # model.add(TimeDistributed(Dense(input_dim = hidden_size, output_dim = vocab_size, activation="softmax")))
    # model.compile(loss='mse', optimizer='adam')

    model = SimpleSeq2seq(input_dim=input_size, hidden_dim=hidden_size, output_length=max_out_seq_len, output_dim=input_size)
    model.compile(loss='mse', optimizer='rmsprop')
    print(model.summary())
    return model

def vectorize_stories(data, word_idx, question_maxlen, answer_maxlen):
    X = []
    Y = []
    for query, answer in data:
        x = [word_idx[w] for w in query]
        y = [word_idx[w] for w in answer]#np.zeros(vocab_size)
        #y[word_idx[answer]] = 1
        X.append(x)
        Y.append(y)
    return pad_sequences(X, maxlen=question_maxlen), pad_sequences(Y, maxlen=answer_maxlen)#np.array(Y)

def get_data(file_path):
    data = pd.read_csv(file_path)
    questions = data.Question.apply(tokenize).values
    answers = data.Answer.apply(tokenize).values

    #questions.apply(tokenize)
    #answers.apply(tokenize)

    data = [(q, a) for q, a in zip(questions, answers)]
    return data

train = get_data("qa_dataset.csv")
#print(train[100:105])
vocab = sorted(reduce(lambda x, y: x | y, (set(question + answer) for question, answer in train)))
# Reserve 0 for masking via pad_sequences
vocab_size = len(vocab) + 1
word_idx = dict((c, i + 1) for i, c in enumerate(vocab))
question_maxlen = max(map(len, (x for x, _ in train)))
answer_maxlen = max(map(len, (x for _, x in train)))

X, Y = vectorize_stories(train, word_idx, question_maxlen, answer_maxlen)

print('vocab = {}'.format(len(vocab)))
print('X.shape = {}'.format(X.shape))
print('Y.shape = {}'.format(Y.shape))
print('question_maxlen, answer_maxlen = {}, {}'.format(question_maxlen, answer_maxlen))

input_size = 64
hidden_size = 128
batch_size = 32
epochs = 20
print('Embed / Hidden Size / Batch size / Epochs = {}, {}, {}, {}'.format(input_size, hidden_size, batch_size, epochs))

X = np.reshape(X, X.shape + (1,))
Y = np.reshape(Y, Y.shape + (1,))
input_shape = X.shape[1:]

print(X.shape, Y.shape)

model = build_model(vocab_size, input_size, max(answer_maxlen,question_maxlen) , hidden_size, input_shape=input_shape)
model.fit([X], Y, batch_size=batch_size, nb_epoch=epochs, validation_split=0.05)