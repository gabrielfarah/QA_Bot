import numpy as np
np.random.seed(1337) # for reproducibility

import pandas as pd
import re

from keras.models import Sequential
from keras.layers.recurrent import LSTM, SimpleRNN
from keras.layers.wrappers import TimeDistributed
from keras.layers.core import Dense, RepeatVector, Activation, Dropout, Flatten
from keras.preprocessing.sequence import pad_sequences
from keras.layers.embeddings import Embedding

from keras.engine.training import slice_X


def tokenize(sent):
    '''Return the tokens of a sentence including punctuation.
    >>> tokenize('Bob dropped the apple. Where is the apple?')
    ['Bob', 'dropped', 'the', 'apple', '.', 'Where', 'is', 'the', 'apple', '?']
    '''
    return [x.strip() for x in re.split('(\W+)?', sent) if x.strip()]

def get_data(file_path):
    data = pd.read_csv(file_path)

    # Aditional step
    #data = data[data.Answer == 'yes' | data.Answer =='no']
    #data = data[data['Answer'].str.strip() == 'yes']
    data = data[data.Answer.str.endswith('yes') | data.Answer.str.endswith('no')]
    data = data[data.Answer.str.len() <= 3]
    data = data[data.Question.str.len() <= 32]

    questions = data.Question.apply(tokenize).values
    answers = data.Answer.apply(tokenize).values

    #questions.apply(tokenize)
    #answers.apply(tokenize)

    data = [(q, a) for q, a in zip(questions, answers)]
    return data

def encode(sentence, maxlen, vocab_size, word_idx):
	X = np.zeros((maxlen, vocab_size))
	for i, c in enumerate(sentence):
		X[i, word_idx[c]] = 1
	return X

def decode(X, word_idx, calc_argmax=True):
	if calc_argmax:
		X = X.argmax(axis=-1)
	return ''.join(word_idx[x] for x in X)

train = get_data("qa_dataset.csv")
#print(train[100:105])
vocab = sorted(reduce(lambda x, y: x | y, (set(question + answer) for question, answer in train)))
# Reserve 0 for masking via pad_sequences
vocab_size = len(vocab) + 1
word_idx = dict((c, i + 1) for i, c in enumerate(vocab))
indices_word = dict((i , c) for i, c in enumerate(vocab))
question_maxlen = max(map(len, (x for x, _ in train)))
answer_maxlen = max(map(len, (x for _, x in train)))

max_question_answer = max(question_maxlen, answer_maxlen)

#PAD?
#questions, answers = 

print('Vectorization...')
X = np.zeros((len(train), max_question_answer, vocab_size), dtype=np.bool)
y = np.zeros((len(train), max_question_answer, vocab_size), dtype=np.bool)

print(X.shape, y.shape)

for i, sentence in enumerate(train[0]): # train[0] == questions
    X[i] = encode(sentence, max_question_answer, vocab_size, word_idx)
for i, sentence in enumerate(train[1]): # train[1] == answers
    y[i] = encode(sentence, max_question_answer, vocab_size, word_idx)

print('vocab = {}'.format(len(vocab)))
print('X.shape = {}'.format(X.shape))
print('Y.shape = {}'.format(y.shape))
print('question_maxlen, answer_maxlen = {}'.format(max_question_answer))

hidden_size = 1024
batch_size = 64
epochs = 200
print('Hidden Size / Batch size / Epochs = {}, {}, {}'.format(hidden_size, batch_size, epochs))

print('Build model...')
model = Sequential()
# "Encode" the input sequence using an RNN, producing an output of HIDDEN_SIZE
# note: in a situation where your input sequences have a variable length,
# use input_shape=(None, nb_feature).
model.add(LSTM(hidden_size, input_shape=(max_question_answer, vocab_size)))
# For the decoder's input, we repeat the encoded input for each time step
model.add(RepeatVector(max_question_answer))
# The decoder LSTM could be multiple layers stacked or a single layer
for _ in range(1):
    model.add(LSTM(hidden_size, return_sequences=True))

# For each of step of the output sequence, decide which character should be chosen
model.add(TimeDistributed(Dense(vocab_size)))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(X, y, batch_size=batch_size, nb_epoch=epochs, validation_split=0.05)

# Explicitly set apart 10% for validation data that we never train over
# split_at = len(X) - len(X) / 10
# (X_train, X_val) = (slice_X(X, 0, split_at), slice_X(X, split_at))
# (y_train, y_val) = (y[:split_at], y[split_at:])

# for iteration in range(1, 200):
#     print('-' * 50)
#     print('Iteration', iteration)
#     model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=1,
#               validation_data=(X_val, y_val))
#     ###
#     # Select 10 samples from the validation set at random so we can visualize errors
#     for i in range(2):
#         ind = np.random.randint(0, len(X_val))
#         rowX, rowy = X_val[np.array([ind])], y_val[np.array([ind])]
#         preds = model.predict_classes(rowX, verbose=0)
#         q = decode(rowX[0], word_idx)
#         correct = decode(rowy[0], word_idx)
#         guess = decode(preds[0], indices_word, calc_argmax=False)
#         print('Q', q)
#         print('T', correct)
#         print('G',guess)
#         print('---')
